# Phân Tích Bóng Đá

App web xem trên điện thoại: lịch thi đấu, sức mạnh đội bóng và phân tích
trước trận bằng mô hình thống kê. Chạy **hoàn toàn miễn phí**, không có
chi phí máy chủ.

---

## Chạy thử ngay

```bash
python3 -m venv .venv && .venv/bin/pip install -r pipeline/requirements.txt
.venv/bin/python pipeline/build.py
python3 -m http.server 8765 --directory public
```

Mở `http://localhost:8765`. **Không cần API key nào** cho Ngoại hạng Anh
và La Liga.

---

## Kiến trúc

```
GitHub Actions (cron, miễn phí)
        │  chạy pipeline/build.py 3 lần/ngày
        ▼
   nguồn dữ liệu  ──►  mô hình Dixon-Coles  ──►  public/data/*.json
        │                                              │
        │                                              │ commit vào repo
        ▼                                              ▼
  openfootball                              Cloudflare Pages (miễn phí)
  football-data.org                                    │
  API-Football                                         ▼
                                              PWA trên điện thoại
```

Điểm mấu chốt: **điện thoại không bao giờ gọi API bóng đá**. Nó chỉ đọc file
JSON tĩnh đã dựng sẵn. Nhờ vậy không bao giờ chạm giới hạn tốc độ của gói
miễn phí, tải gần như tức thì, và xem được offline.

Tổng dung lượng dữ liệu: khoảng 160 KB cho 2 giải.

---

## Nguồn dữ liệu

| Nguồn | Dùng cho | Cần key | Giới hạn |
|---|---|---|---|
| **openfootball** | Kết quả lịch sử, lịch thi đấu (12 giải) | Không | Không |
| **football-data.org** | Champions League + kết quả mới các giải | Có (miễn phí) | 10 req/phút, 12 giải |
| **The Odds API** | Kèo trận sắp tới | Có (miễn phí) | 500 lượt/tháng |
| **ESPN** | Người ghi bàn, tỷ số hiệp 1 | Không | Không (API không tài liệu) |
| **mirror football-data** | Kèo đóng cửa lịch sử | Không | Không |

Đăng ký key miễn phí (tuỳ chọn):
- football-data.org — https://www.football-data.org/client/register
- The Odds API — https://the-odds-api.com/

Copy `.env.example` thành `.env` rồi điền key vào.

> **Lưu ý về độ bền:** ngay trong lúc dựng dự án này, football-data.co.uk
> trả lỗi 503 và ClubElo không phản hồi. Vì thế xương sống dữ liệu đặt ở
> openfootball — phục vụ qua hạ tầng GitHub nên gần như không bao giờ sập.
>
> Kèo lịch sử lấy từ một bản mirror GitHub của football-data.co.uk. Vì đó là
> repo bên thứ ba không rõ danh tính, `odds_history.py` đối chiếu kết quả trận
> đấu với openfootball trước khi dùng — lần kiểm tra gần nhất khớp 380/380 trận
> và biên nhà cái trung bình 2.95%, đúng đặc trưng của Pinnacle.

---

## Mô hình dự đoán

**Dixon-Coles (1997)** — Poisson hai biến có hiệu chỉnh tỷ số thấp.

Mỗi đội có hai tham số: sức tấn công và sức phòng ngự. Với trận đấu giữa
đội nhà *i* và đội khách *j*:

```
λ = exp(intercept + atk[i] + def[j] + lợi_thế_sân_nhà)
μ = exp(intercept + atk[j] + def[i])

P(x, y) = tau(x, y, λ, μ) · Poisson(x; λ) · Poisson(y; μ)
```

`tau` sửa lại xác suất cho bốn tỷ số thấp (0-0, 0-1, 1-0, 1-1) — nơi mô hình
Poisson độc lập lệch rõ so với thực tế.

Từ ma trận tỷ số đầy đủ, mọi thị trường (1X2, tài/xỉu, cả hai đội ghi bàn,
tỷ số chính xác) đều được suy ra, nên chúng luôn nhất quán với nhau.

### Hai kỹ thuật quan trọng

**Suy giảm theo thời gian** (`xi`) — trận cũ có trọng số nhỏ hơn:
`w = exp(-xi · số_ngày_trước)`. Nhờ đó phong độ gần đây được coi trọng hơn.

**Co ngót Bayes** (`ridge`) — kéo sức mạnh của đội ít dữ liệu về mức trung
bình giải. Không có bước này, một đội mới thăng hạng thắng 2 trận đầu mùa
sẽ bị mô hình gán hệ số phòng ngự `-5.24`, tức là gần như không thể thủng
lưới. Có co ngót, con số đó về mức hợp lý.

### Lớp hiệu chỉnh bối cảnh

Nhân vào λ và μ. Hiện đang hoạt động: **ngày nghỉ** giữa hai trận, và **động
lực** theo vị trí trên bảng xếp hạng ở giai đoạn cuối mùa.

Lớp này còn hỗ trợ tính cả cầu thủ vắng mặt (theo vị trí, số phút thường đá và
mức quan trọng, có bão hoà mềm để một ca vắng mặt không làm lệch dự đoán quá
mức), nhưng **đầu vào đó hiện không được dùng**: không có nguồn dữ liệu chấn
thương miễn phí nào cho mùa đang diễn ra. Mã được giữ nguyên trong
`model/adjustments.py`, sẵn sàng nối lại nếu sau này có nguồn.

Các hệ số Dixon-Coles được **ước lượng từ dữ liệu**. Các hệ số hiệu chỉnh thì
**không** — chúng là suy đoán có cơ sở, đặt ở mức dè dặt. Mọi hiệu chỉnh đều
bị chặn biên độ, và app luôn hiển thị đóng góp của từng yếu tố thay vì đưa ra
một con số duy nhất không giải thích được.

---

## Vì sao app không có ô nhập API key

Trang web chỉ đọc file JSON dựng sẵn — nó **không bao giờ tự gọi API bóng đá**.
Người gọi API là pipeline Python. Nếu đặt key vào giao diện thì key nằm trong
mã trang mà bất kỳ ai mở link đều đọc được, và mỗi người xem lại tiêu quota
của bạn.

Key đặt ở hai chỗ:

- **Chạy trên máy** — file `.env`
- **Chạy trên GitHub** — Settings → Secrets and variables → Actions

Tab **Cài đặt** trong app hiển thị nguồn nào đang hoạt động, nguồn nào bị chặn,
dữ liệu cập nhật lần cuối lúc nào — nhưng chỉ hiện trạng thái có/không, không
bao giờ hiện giá trị key.

---

## Lịch sử và đối chiếu dự đoán

`predictions.json` chỉ chứa trận SẮP tới và bị ghi lại mỗi lần build — trận đá
xong là dự đoán biến mất. Không lưu lại thì không thể trả lời câu hỏi quan
trọng nhất: **mô hình này đoán đúng bao nhiêu phần?**

`pipeline/history.py` chạy trước khi ghi đè: đọc bản của lần build trước, tìm
trận nay đã đá, gắn kết quả thật cùng người ghi bàn, rồi nối vào `history.json`.

Bản dự đoán được lưu là bản của lần build **gần nhất trước giờ bóng lăn**, chứ
không phải bản đầu tiên — đó mới là dự đoán người dùng thực sự nhìn thấy.

Mỗi trận được chấm ba tiêu chí:

- **Đúng cửa** — cửa app nêu trong phần nhận định có trúng không
- **Xác suất gán cho kết quả thật** — quan trọng hơn đúng/sai. Trúng với 70%
  đáng khen hơn trúng với 40%; trượt khi đã gán 15% cho kết quả thật thì tệ
  hơn nhiều so với trượt khi gán 33%
- **Brier** cho từng trận, cộng lại thành điểm trung bình

Người ghi bàn lấy từ API công khai của ESPN (`sources/espn.py`) — kèm phút,
người kiến tạo, và tỷ số hiệp 1 suy từ các bàn thuộc hiệp 1. Gói miễn phí của
football-data.org không trả mảng `goals`, còn API-Football chặn mùa hiện tại.

`pipeline/backfill_history.py` khôi phục lịch sử từ các phiên bản
`predictions.json` cũ trong git — đã cứu được 8 trận có dự đoán đưa ra trước
khi hệ thống lưu trữ tồn tại.

---

## Kèo hiệp 1

Suy từ bàn thắng kỳ vọng cả trận nhân với tỷ lệ bàn ghi trong hiệp 1, đo từ
chính dữ liệu của giải: **44.5%** ở Ngoại hạng Anh (1.821 trận), **44.1%** ở
La Liga (1.783 trận), **44.2%** trên 5 giải lớn (5.000 trận).

Đối chiếu với kèo thật của nhà cái cho trận Aston Villa vs Nottingham Forest:

| | nhà | hoà | khách | tài 0.5 H1 |
|---|---|---|---|---|
| nhà cái | 33.9% | 42.2% | 23.9% | 67.1% |
| mô hình | 32.5% | 42.9% | 24.6% | 68.3% |
| lệch | −1.4 | +0.7 | +0.7 | +1.1 |

Lệch tối đa 1.4 điểm phần trăm. Tự tính thay vì mua kèo vì thị trường hiệp 1
chỉ có qua endpoint riêng từng trận của The Odds API, tốn 4 lượt mỗi trận —
với 61 trận thì vượt xa trần 500 lượt/tháng.

### Vì sao không có dự đoán người ghi bàn

Đã dựng thử và **loại bỏ sau khi đối chiếu**. Dữ liệu miễn phí duy nhất là
endpoint scorers của football-data.org: 60 cầu thủ toàn giải, chỉ gồm những
người ĐÃ ghi bàn. Nhà cái treo kèo cho 35 cầu thủ mỗi trận, gồm cả người chưa
ghi bàn nào.

Kết quả khi thử: Aston Villa chưa ghi bàn nào sau 3 vòng nên không có cầu thủ
nào để tính; Tammy Abraham — cửa nhà cái đánh giá cao nhất — không có trong dữ
liệu; sai số so với kèo thật là **8-11 điểm phần trăm**.

Người ghi bàn THỰC TẾ của các trận đã đá thì có đầy đủ trong phần lịch sử.

---

## Đọc kết quả: cửa nào đáng tin

Một xác suất trần trụi không trả lời được câu hỏi "nên tin cái gì". App tính
thêm **mức khác thường** cho từng cửa:

```
mức khác thường = xác suất của trận này − tần suất cửa đó ở một trận bất kỳ
```

Tần suất nền lấy từ chính lịch sử của giải đó, không phải hằng số chung —
La Liga hoà nhiều hơn Ngoại hạng Anh, Bundesliga nhiều bàn hơn Serie A.

Bước này quan trọng hơn vẻ ngoài của nó. Ban đầu tôi so với mốc ngẫu nhiên
(1/3 cho 1X2, 1/2 cho hai cửa) và kết quả là **"Tài 1.5" đứng đầu ở gần như
mọi trận** — đúng, vì nó xảy ra ~75% ở mọi trận bóng đá, nhưng nó chẳng nói
lên điều gì riêng về trận đang xét. So với nền thực tế thì "Tài 1.5 86%, nền
80%" chỉ vượt 6 điểm và tự động rơi xuống cuối bảng.

Mỗi trận được xếp một bậc — **rõ / vừa / mơ hồ** — và bị hạ bậc khi mô hình
lệch thị trường quá 8 điểm phần trăm, khi dữ liệu đội bóng mỏng, hoặc khi
chưa có kèo. Phân bố thực tế trên 69 trận: 36 mơ hồ, 28 vừa, 5 rõ. Phần lớn
trận bóng đá **không** có tín hiệu rõ, và app nói thẳng điều đó thay vì bịa ra
một con số nghe chắc chắn.

Mỗi kết luận đều kèm tỷ lệ hỏng: "vẫn hỏng khoảng 34% số lần".

---

## Kết quả kiểm định

Kiểm định cuốn chiếu trên Ngoại hạng Anh, 780 trận (mùa 2024-25 và 2025-26).
Khi dự đoán mỗi trận, mô hình chỉ được nhìn các trận đã đá **trước** ngày đó.

| | RPS ↓ | Log-loss ↓ | Chính xác ↑ |
|---|---|---|---|
| **Mô hình Dixon-Coles** | **0.2069** | **1.0097** | **50.1%** |
| Tần suất lịch sử | 0.2315 | 1.0829 | 41.9% |
| Đoán đều 1/3 | 0.2348 | 1.0986 | 41.9% |

Hiệu chuẩn tốt: trong nhóm dự đoán "khoảng 45%", thực tế xảy ra 45.7%;
nhóm "khoảng 55%" thực tế 55.8%.

### Chọn tham số — và vì sao tinh chỉnh không giúp được nhiều

Dò lưới 5×6 trên `xi` và `ridge` (1.560 trận, EPL + La Liga):

- `ridge = 2.0` tốt nhất ở **cả 5** mức `xi` — tín hiệu nhất quán về hướng.
- `xi = 0` (không suy giảm thời gian) luôn tệ nhất → phong độ gần đây có ý nghĩa.
- `ridge = 30` luôn tệ → co ngót quá tay xoá mất khác biệt giữa các đội.

Quán quân dò lưới là `xi=0.0030, ridge=2.0` (RPS 0.20172). Nhưng khi **kiểm
chứng trên giai đoạn 2022-2024 mà dò lưới chưa từng thấy**, thứ tự đảo lại:

| cấu hình | RPS trên giai đoạn giữ riêng |
|---|---|
| `xi=0.0018, ridge=2.0` | **0.19823** |
| `xi=0.0030, ridge=2.0` | 0.19845 |
| `xi=0.0018, ridge=8.0` (mặc định cũ) | 0.19905 |

Quán quân dò lưới tụt xuống thứ hai — dấu hiệu điển hình của khớp quá mức vào
chính tập dùng để chọn nó.

Quan trọng hơn: **bootstrap theo cặp cho thấy không cặp nào khác biệt có ý
nghĩa thống kê** (mọi khoảng tin cậy 95% đều chứa 0). Việc tinh chỉnh không
mang lại cải thiện chứng minh được. Dự án chốt `xi=0.0018, ridge=2.0` vì đó là
cấu hình có bằng chứng nhất quán nhất và không có mặt trái, chứ không phải vì
nó "thắng".

Đây cũng là lời nhắc: phần lớn giá trị của mô hình nằm ở việc chọn đúng dạng
mô hình và xử lý đúng đội thiếu dữ liệu, không phải ở việc vặn tham số.

Chạy lại:

```bash
.venv/bin/python pipeline/backtest.py --leagues premier-league --test-start 2024-08-01
.venv/bin/python pipeline/backtest.py --grid          # dò tham số xi và ridge
```

---

## Cập nhật kết quả sau mỗi vòng

Hai nguồn, mỗi nguồn làm đúng việc nó giỏi:

| nguồn | vai trò | độ trễ | giới hạn |
|---|---|---|---|
| openfootball | lịch sử sâu (20+ mùa) | ~1 tuần | không có |
| football-data.org | kết quả mùa hiện tại | vài phút sau trận | 10 req/**phút** |

`sources/results.py` nạp lịch sử từ openfootball rồi bổ sung các trận mới từ
football-data.org, loại trùng theo (ngày, đội nhà, đội khách). Tên đội của
football-data.org được quy về tên openfootball trước khi so, nếu không
"Deportivo Alavés" và "Alavés" sẽ thành hai đội riêng biệt trong mô hình.

Nhờ vậy bảng xếp hạng, cột thắng/hoà/thua và phong độ 5 trận cập nhật trong
vòng vài giờ sau trận đấu, thay vì đợi tới bảy ngày.

### Lịch chạy và cam kết 6 tiếng

Hai workflow, tách theo mức nặng nhẹ:

| | nhịp | thời lượng | làm gì |
|---|---|---|---|
| `refresh-live.yml` | mỗi **3 tiếng** | ~2 phút | bảng xếp hạng, kết quả, lịch sử, đội hình |
| `update-data.yml` | mỗi **8 tiếng** | ~7 phút | khớp lại mô hình, sinh dự đoán, lấy kèo |

**Vì sao 3 tiếng khi chỉ cần đảm bảo 6:** cron của GitHub Actions gói miễn phí
là "cố gắng" chứ không cam kết. Đo trên chính repo này, các lần chạy theo lịch
bị trễ **6, 8, 9, 24, 30 và 37 phút**. Đặt đúng 6 tiếng thì chỉ cần một lần trễ
hoặc bị bỏ qua là vượt mốc. Đặt 3 tiếng thì kể cả mất trọn một nhịp vẫn còn
trong hạn.

Ngân sách: 8 × 2 + 3 × 7 = 37 phút/ngày = **1.110 phút/tháng**, trong hạn 2.000.

Hai loại dữ liệu có ràng buộc quota khác nhau nên tách riêng:

- **Kết quả trận** — football-data.org giới hạn 10 request/*phút*, không giới
  hạn theo ngày. Chạy bao nhiêu cũng được.
- **Kèo** — The Odds API 500 lượt/*tháng*, mỗi giải tốn 2 lượt. Tự hãm bằng
  cache 11 tiếng nên chỉ thực sự gọi khoảng 2 lần/ngày.

App hiển thị đếm ngược tới lần cập nhật kế tiếp, tính từ lịch ghi trong
`index.json` cộng 25 phút đệm cho độ trễ thường gặp.

---

## Champions League — mô hình phân tầng

C1 không dùng được mô hình một giải: mỗi CLB chỉ đá ~8 trận vòng bảng mỗi mùa,
quá ít để ước lượng sức mạnh. Nhưng gộp thẳng nhiều giải vào một mô hình lại
cho kết quả sai — thử lần đầu, mô hình xếp **Celtic hạng 7 châu Âu, trên cả
Manchester City**. Lý do: Celtic đá 38 trận thắng đậm ở Scotland và chỉ 8 trận
C1, nên hàm hợp lý bị phần quốc nội áp đảo, mà mô hình lại không có cách nào
biết giải Scotland yếu hơn.

Lời giải là tách sức mạnh thành hai tầng:

```
sức_mạnh_thật[i] = atk[i] + league_atk[giải_của_i]
```

- `atk[i]` là sức mạnh **trong giải**, chuẩn hoá tổng bằng 0 ở từng giải.
  Xác định rất chắc nhờ hàng trăm trận quốc nội.
- `league_atk[L]` là **mặt bằng cả giải**. Các trận trong cùng một giải hoàn
  toàn không mang thông tin về đại lượng này (nó triệt tiêu giữa hai đội cùng
  giải), nên nó chỉ được xác định bởi **các trận C1 đóng vai trò cầu nối**.

Kết quả sau khi sửa: Celtic rơi từ hạng 7 xuống **hạng 55**, Olympiakos 8→24,
Galatasaray 12→40. Top đầu thành Arsenal, PSG, Manchester City, Bayern — hợp lý.

Mặt bằng giải mô hình tự học ra (cao = mạnh):

| giải | mức | | giải | mức |
|---|---|---|---|---|
| Ngoại hạng Anh | +0.79 | | Bồ Đào Nha | −0.21 |
| Ligue 1 | +0.59 | | Bỉ | −0.24 |
| La Liga | +0.34 | | Thổ Nhĩ Kỳ | −0.27 |
| Bundesliga | +0.26 | | Hà Lan | −0.33 |
| Serie A | +0.08 | | Scotland | −0.44 |
| Hy Lạp | 0.00 | | Áo | −0.52 |

Dữ liệu: 9.514 trận từ 12 giải quốc nội, với 378 trận C1 hai mùa gần nhất làm
cầu nối. Mô hình có ~580 tham số nên cần `max_iter=8000` mới hội tụ thật.

### Độ tin cậy phân biệt được hai tình huống

Cách tính ngây thơ gộp làm một hai trường hợp rất khác nhau:

- **Viking FK** — không có dữ liệu quốc nội nào (openfootball không có giải
  Na Uy). Mô hình mặc định coi là "trung bình châu Âu", gần như chắc chắn quá
  cao. Đánh dấu **thấp**, và pipeline in cảnh báo tên CLB ra log.
- **Porto** — 106 trận quốc nội nhưng 0 trận C1 hai mùa qua. Vị trí trên thang
  châu Âu vẫn suy được từ mặt bằng giải Bồ Đào Nha, vốn đã hiệu chỉnh nhờ
  Sporting và Benfica (44 trận cầu nối). Đánh dấu **trung bình**, không phải thấp.

---

## Kết hợp kèo nhà cái

Kèo đóng cửa được xử lý qua ba bước:

**1. Khử biên nhà cái.** Kèo luôn cộng biên lợi nhuận nên tổng nghịch đảo tỷ lệ
lớn hơn 1. Có hai cách khử: chia đều theo tỷ lệ, và phương pháp Shin (mô hình
hoá biên như hệ quả của người đặt cược có thông tin nội bộ). Shin nâng cửa trên
và hạ cửa dưới ở các trận chênh lệch lớn — đúng phần thiên lệch cần sửa.

**2. Ép mô hình khớp 1X2 thị trường.** Kèo chỉ cho ba con số, nhưng app còn cần
tỷ số chính xác, tài/xỉu, BTTS. Nên thay vì lấy 1X2 từ kèo và phần còn lại từ
mô hình (sẽ mâu thuẫn nhau), pipeline giải ngược ra cặp (λ, μ) tạo đúng 1X2 đó,
rồi suy mọi thị trường từ ma trận đã hiệu chỉnh. Sai số khớp ở mức chính xác máy.

**2b. Ghim luôn tổng bàn.** Chỉ khớp 1X2 là chưa đủ: nó ràng buộc *hiệu số*
bàn thắng, nhưng hai cặp (λ, μ) rất khác nhau như (2.4, 0.9) và (1.6, 0.4) có
thể cho cùng xác suất thắng/hoà/thua mà phân phối tỷ số khác hẳn. Nên pipeline
lấy thêm kèo tài/xỉu và giải **ba ẩn (λ, μ, rho)** khớp cả 1X2 lẫn tổng bàn.

Khác biệt không nhỏ. Chelsea gặp Hull City, thị trường định giá tài 2.5 là
66.6%:

| | tài 2.5 mô hình cho | ba tỷ số hàng đầu |
|---|---|---|
| chỉ khớp 1X2 | 56.3% (lệch **−10.3 điểm**) | 2-0, 1-0, 3-0 |
| khớp cả tổng bàn | 66.6% (lệch 0.0) | 2-0, 3-0, 2-1 |

Bản hai ẩn cho toàn tỷ số sạch lưới và tổng bàn thấp giả tạo. Sai số của bản
ba ẩn ở mức 10⁻⁸.

**3. So lệch.** Xác suất **thuần mô hình** đem so với kèo để tìm chỗ hai bên bất
đồng. Phải dùng bản thuần, không dùng bản đã hợp nhất — hợp nhất kéo ước lượng
về phía thị trường nên tự triệt tiêu chính chênh lệch đang muốn tìm.

### Kết quả kiểm định — và một kết luận không như mong đợi

2.270 trận Ngoại hạng Anh + La Liga, kèo đóng cửa Pinnacle:

| | RPS ↓ | Chính xác ↑ |
|---|---|---|
| **Thị trường thuần** | **0.19147** | **54.9%** |
| Hợp nhất w=0.9 | 0.19172 | 54.9% |
| Hợp nhất w=0.8 | 0.19208 | 54.8% |
| Mô hình thuần | 0.19912 | 51.2% |

**Trọng số tối ưu là w = 1.0, tức thị trường thuần.** Trộn thêm mô hình vào làm
dự báo *tệ đi*, và bootstrap theo cặp xác nhận chênh lệch này có ý nghĩa thống
kê (KTC 95% không chứa 0), chứ không phải nhiễu mẫu.

Nói cách khác: mô hình không mang thêm thông tin nào mà giá thị trường chưa có.
Đây là kết quả đã được ghi nhận nhiều trong nghiên cứu — kèo đóng cửa cực kỳ
hiệu quả vì nó gói cả tin đội hình phút chót mà mô hình không thể biết.

**Vì thế app dùng thị trường làm con số chính khi có kèo** (54.9% so với 51.2%
là cải thiện đáng kể), còn mô hình giữ ba vai trò riêng mà kèo không làm được:
giải thích *vì sao* (bàn thắng kỳ vọng, sức mạnh đội hình, ca vắng mặt), dự đoán
các trận không có kèo, và chỉ ra chỗ bất đồng.

Chạy lại:

```bash
.venv/bin/python pipeline/blend_backtest.py --leagues premier-league la-liga --test-start 2023-08-01
```

### Nói thẳng về giới hạn

- Mô hình **không đánh bại được nhà cái** một cách hệ thống. Kèo đóng cửa
  thường đạt RPS khoảng 0.185–0.190, tốt hơn con số 0.2069 ở trên. Kèo gói
  cả thông tin nội bộ mà mô hình không thể biết.
- Độ chính xác 50% nghĩa là **sai một nửa số lần**. Bóng đá có phương sai rất
  lớn và điều đó không thể khắc phục bằng mô hình tốt hơn.
- Giá trị thật của app là phân tích có cấu trúc và chỉ ra chỗ mô hình lệch
  thị trường — không phải công cụ kiếm tiền.

---

## Triển khai miễn phí

1. Đẩy repo lên GitHub.
2. Vào Cloudflare Pages → Connect to Git → chọn repo.
   - Build command: *(để trống)*
   - Build output directory: `public`
3. Trong GitHub → Settings → Secrets and variables → Actions, thêm key:
   `ODDS_API_KEY` (quan trọng nhất), `FOOTBALL_DATA_ORG_KEY`, `API_FOOTBALL_KEY`.

   Đừng bao giờ đưa key vào file được commit. `.env` đã nằm trong `.gitignore`;
   trên CI thì key đến từ Secrets và `config.py` ưu tiên biến môi trường sẵn có
   nên không cần `.env`.

GitHub Actions sẽ tự cập nhật dữ liệu 3 lần mỗi ngày và commit vào repo,
Cloudflare Pages tự triển khai lại.

Trên điện thoại, mở link rồi chọn **Thêm vào màn hình chính** để dùng như
app thật, có biểu tượng riêng và xem được offline.

**Chi phí thật sự: 0 đồng.** Cloudflare Pages miễn phí không giới hạn băng
thông; GitHub Actions miễn phí 2.000 phút/tháng (pipeline dùng khoảng 30
phút/tháng).

---

## Cấu trúc dự án

```
pipeline/
  build.py               điều phối: nạp dữ liệu -> khớp mô hình -> ghi JSON
  backtest.py            kiểm định cuốn chiếu và dò tham số
  model/
    dixon_coles.py       mô hình lõi
    adjustments.py       hiệu chỉnh ngày nghỉ / động lực (và vắng mặt, chưa dùng)
    markets.py           ma trận tỷ số -> 1X2, tài xỉu, BTTS
    evaluate.py          RPS, Brier, log-loss, hiệu chuẩn
    predictor.py         gộp thành một dự đoán hoàn chỉnh
  history.py             lưu dự đoán và chấm điểm sau trận
  backfill_history.py    khôi phục lịch sử từ git (chạy một lần)
  model/
    cross_league.py      mô hình phân tầng cho Champions League
    extra_markets.py     kèo hiệp 1
  sources/
    espn.py              người ghi bàn và tỷ số hiệp 1
    openfootball.py      kết quả và lịch 12 giải (không cần key)
    football_data_org.py lịch, kết quả và BXH Champions League
    european.py          gộp dữ liệu quốc nội + C1 làm cầu nối
    odds_history.py      kèo đóng cửa lịch sử + ghép tên CLB
    odds_live.py         kèo trận sắp tới (The Odds API)
    historical.py        football-data.co.uk (nguồn gốc, hay sập)
public/                  PWA — HTML/CSS/JS thuần, không cần build
  data/                  JSON do pipeline sinh ra
```

---

## Bước tự kiểm tra

`pipeline/selfcheck.py` chạy sau mỗi lần build và chặn việc xuất bản nếu có lỗi:

- **Hàm JavaScript bắt buộc** — 23 hàm phải tồn tại trong `app.js`
- **JSON đúng chuẩn** — từ chối cả `NaN` và `Infinity`, thứ mà `json.load` của
  Python vẫn đọc được nhưng trình duyệt thì không
- **Dự đoán đủ trường** — và xác suất 1X2 phải cộng lại bằng 1

Lý do có bước này rất cụ thể: hai lần liên tiếp tôi sửa `app.js` bằng cắt ghép
theo vị trí chuỗi và vô tình xoá mất cả hàm — lần đầu mất `matchCard` và
`renderFixtures`, lần sau mất `marketSection`. Không lần nào bị phát hiện lúc
build; cả hai lần đều do người dùng bấm vào app rồi báo hỏng.

Bộ kiểm tra thoát với mã 1 khi có lỗi, nên GitHub Actions cũng dừng thay vì
đẩy bản hỏng lên.

```bash
.venv/bin/python pipeline/selfcheck.py
```

---

## Ba lỗi âm thầm đã tìm ra khi chạy thật

Cả ba đều không báo lỗi ở đâu cả — chúng chỉ làm app sai một cách lặng lẽ.

**1. Kết quả không bao giờ cập nhật.** Có **hai** nguyên nhân chồng lên nhau,
và sửa cái thứ nhất không đủ.

*Nguyên nhân gần:* cache openfootball không có hạn dùng, nên mùa đang diễn ra
tải một lần rồi giữ mãi. Nay mùa đã kết thúc thì cache vĩnh viễn (kết quả
không đổi được nữa), mùa đang diễn ra có hạn 3 giờ.

*Nguyên nhân gốc:* **bản thân openfootball chỉ cập nhật khoảng một lần mỗi
tuần** — lịch sử commit ghi rõ "auto-update week 36", "week 35". Ngày 7/9 mà
kết quả mới nhất vẫn là 31/8, toàn bộ vòng đấu cuối tuần chưa có. Chạy pipeline
dày hơn cũng vô ích vì nguồn chưa có dữ liệu.

**2. Tab La Liga trắng trơn.** Phần lớn trận La Liga không có giờ thi đấu, nên
trường `time` là `NaN`. Python ghi thẳng `NaN` vào JSON — hợp lệ với Python
nhưng **không hợp lệ với JSON**, và `JSON.parse` của trình duyệt từ chối cả
file. Ngoại hạng Anh không dính vì trận nào cũng có giờ. Nay mọi NaN được dọn
thành `null`, và `allow_nan=False` biến lỗi tương lai thành lỗi ồn ào lúc build.

**3. Bản cập nhật mã không tới được người dùng.** Service worker trả bản cache
rồi tải bản mới ở nền — nhưng thiếu `event.waitUntil()` nên trình duyệt tắt
worker trước khi ghi xong, và lệnh fetch nền lại lấy từ HTTP cache nên ghi đè
bằng chính bản cũ. Nhãn tab vẫn hiện tên cũ sau ba lần tải lại. Nay có
`waitUntil` cộng `cache: "no-cache"` ở lúc làm mới và `cache: "reload"` ở lúc
cài đặt.

---

## V-League: vì sao không có trong app

Đã thử sáu nguồn, không nguồn nào đủ dùng:

| nguồn | kết quả |
|---|---|
| openfootball | Không có Việt Nam |
| The Odds API | Không có V-League (chỉ J-League, K-League, Trung Quốc, Úc) |
| football-data.org | Gói free chỉ 12 giải châu Âu |
| ESPN (endpoint ẩn) | HTTP 400, không có giải Việt Nam |
| Wikipedia | Có bài mùa 2026–27 nhưng bảng dữ liệu còn trống |
| API-Football | Gói Free chỉ cho mùa 2023-24, chặn 2025/2026/2027 |

Rào cản không nằm ở lịch thi đấu — trang VPF hoàn toàn scrape được. Rào cản là
**dữ liệu lịch sử để mô hình học**: thứ duy nhất lấy được là 183 trận từ
10/2023 đến 07/2024 với 15 đội, trong khi mùa hiện tại đã có Bắc Ninh, PVF-CAND,
Ninh Bình — những CLB không hề có trong tập đó.

Dựng mô hình trên nền ấy sẽ cho ra các con số trông đáng tin nhưng không có cơ
sở. Đúng loại lỗi âm thầm mà toàn bộ dự án này được thiết kế để tránh, nên
V-League bị bỏ thay vì làm nửa vời.

Muốn thêm sau này thì cần gói trả phí của API-Football (khoảng 19 USD/tháng),
hoặc đợi có nguồn mở phủ được giải này.

---

## Còn thiếu gì

- **V-League** — đã khảo sát và quyết định không thêm. Chi tiết ở mục dưới.
- **Sabah FK, Viking FK** — hai CLB dự C1 từ giải openfootball không có
  (Azerbaijan, Na Uy). Cần thêm nguồn quốc nội cho họ, hoặc chấp nhận nhãn
  tin cậy thấp.
- **xG** — Understat chỉ có 5 giải hàng đầu châu Âu, phải scrape. Dùng xG
  thay bàn thắng thường cho ước lượng ổn định hơn.
- **Kèo trận sắp tới** — đã viết xong client The Odds API, chỉ cần key miễn phí
  để bật.
- **Chấn thương** — không có nguồn miễn phí nào cho mùa hiện tại, nên tính năng
  này đã gỡ. Lớp xử lý trong `model/adjustments.py` vẫn giữ, sẵn sàng nối lại
  nếu sau này có nguồn.
