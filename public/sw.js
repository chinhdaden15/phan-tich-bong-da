/**
 * Service worker cho chế độ offline.
 *
 * Chiến lược:
 *   - Vỏ app (HTML/CSS/JS): stale-while-revalidate. Trả bản cache ngay cho
 *     nhanh, đồng thời tải bản mới ở nền để lần mở sau đã là bản mới.
 *   - Dữ liệu JSON: network-first, dự phòng cache khi mất mạng.
 *
 * KHÔNG dùng cache-first thuần cho vỏ app. Cache-first giữ mãi bản cũ cho tới
 * khi đổi số VERSION bằng tay — và quên đổi một lần là người dùng chạy mã cũ
 * vô thời hạn. Lỗi đó đã xảy ra ngay trong lúc dựng dự án này: giao diện vẫn
 * hiện giờ sân vận động sau khi mã đã sửa sang giờ địa phương.
 */

const VERSION = "v1";
const SHELL_CACHE = `shell-${VERSION}`;
const DATA_CACHE = `data-${VERSION}`;
const CREST_CACHE = `crests-${VERSION}`;

const SHELL = ["./", "index.html", "styles.css", "app.js", "manifest.json", "icon.svg"];

self.addEventListener("install", (event) => {
  // cache.addAll() lấy qua HTTP cache của trình duyệt, nên có thể nạp đúng
  // bản cũ vừa được thay. Tải tường minh với cache:"reload" để luôn lấy bản
  // mới nhất từ máy chủ.
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then((cache) => Promise.all(
        SHELL.map((url) =>
          fetch(new Request(url, { cache: "reload" }))
            .then((response) => {
              if (response && response.ok) return cache.put(url, response);
            })
            .catch(() => undefined)   // thiếu một file không được chặn cài đặt
        )
      ))
      .then(() => precacheCrests())
      .then(() => self.skipWaiting())
  );
});

/**
 * Tải sẵn toàn bộ logo đội và cờ ngay lúc cài đặt.
 *
 * Trước đây ảnh chỉ được tải khi cuộn tới, mà lại nằm trên máy chủ của bên
 * thứ ba nên không cache được — logo hiện chậm, nhấp nháy, phải vào vài lần
 * mới thấy đủ. Nay tải một lượt 120 tệp (~490 KB) rồi thôi.
 *
 * Hỏng thì bỏ qua, KHÔNG chặn việc cài đặt: thiếu logo chỉ là xấu, còn service
 * worker không cài được thì mất luôn chế độ offline.
 */
function precacheCrests() {
  return fetch("crests/manifest.json", { cache: "reload" })
    .then((response) => (response.ok ? response.json() : []))
    .then((files) =>
      caches.open(CREST_CACHE).then((cache) =>
        Promise.all(files.map((name) =>
          cache.match(`crests/${name}`).then((hit) =>
            hit || fetch(`crests/${name}`)
              .then((r) => (r && r.ok ? cache.put(`crests/${name}`, r) : undefined))
              .catch(() => undefined)
          )
        ))
      )
    )
    .catch(() => undefined);
}

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => ![SHELL_CACHE, DATA_CACHE, CREST_CACHE].includes(k))
            .map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Logo đội và cờ: cache trước, KHÔNG kiểm tra lại máy chủ.
  //
  // Tên tệp là mã băm của URL nguồn, nên nội dung một tệp không bao giờ đổi —
  // đội thay logo sẽ ra tên tệp khác. Đã có trong cache thì trả luôn, khỏi tốn
  // một request. Quan trọng vì tab Nations League có tới 176 ô logo: kiểm tra
  // lại từng cái mỗi lần mở app chính là thứ làm logo hiện chậm và nhấp nháy.
  if (url.pathname.includes("/crests/")) {
    event.respondWith(
      caches.open(CREST_CACHE).then((cache) =>
        cache.match(request).then((cached) => cached || fetch(request).then((response) => {
          if (response && response.status === 200) cache.put(request, response.clone());
          return response;
        }))
      )
    );
    return;
  }

  // Dữ liệu: ưu tiên mạng, mất mạng thì lấy bản đã lưu.
  if (url.pathname.includes("/data/")) {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(DATA_CACHE).then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Vỏ app: trả cache ngay, đồng thời làm mới ở nền.
  //
  // event.waitUntil(revalidate) là BẮT BUỘC, không phải tuỳ chọn. Khi có bản
  // cache, respondWith() hoàn tất ngay và trình duyệt được phép tắt service
  // worker — lệnh fetch chạy nền bị huỷ giữa chừng và cache không bao giờ
  // được ghi đè. Hệ quả: mọi bản sửa mã đều không đến được người dùng, kể cả
  // sau nhiều lần tải lại. Đúng lỗi này đã xảy ra: nhãn tab vẫn hiện "Mô hình"
  // sau ba lần reload trong khi máy chủ đã trả về "Cài đặt".
  event.respondWith(
    caches.open(SHELL_CACHE).then((cache) =>
      cache.match(request).then((cached) => {
        // cache:"no-cache" buộc gửi request lên máy chủ để kiểm tra thay vì
        // lấy thẳng từ HTTP cache. Không có nó, lệnh làm mới chỉ đọc lại đúng
        // bản cũ trong HTTP cache rồi ghi đè cache của service worker bằng
        // chính nội dung cũ đó — cập nhật không bao giờ tới nơi.
        const revalidate = fetch(new Request(request.url, { cache: "no-cache" }))
          .then((response) => {
            if (response && response.status === 200 && response.type === "basic") {
              cache.put(request, response.clone());
            }
            return response;
          })
          .catch(() => cached);

        // Giữ service worker sống cho tới khi ghi cache xong.
        event.waitUntil(revalidate);
        return cached || revalidate;
      })
    )
  );
});
