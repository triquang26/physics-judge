# Reader keypoint — đặc tả toán học

DINOv3 đóng băng mã hoá ảnh; một diffusion head 19.8 M tham số khử nhiễu toạ độ
keypoint 3-D có điều kiện trên patch token.

## 0. Ký hiệu

| ký hiệu | nghĩa | giá trị |
|---|---|---|
| `B, T` | batch, số frame trong cửa sổ | 32, 16 (train) |
| `V, P₀, P` | số view; token/view trước pool; sau pool | 1–3, 2304, 576 |
| `D` | bề rộng embed backbone | 1024 |
| `d` | bề rộng token trong head | 512 |
| `h, d_h` | số head attention, bề rộng mỗi head | 8, 64 |
| `K` | số keypoint (theo robot) | 4 / 18 / 22 |
| `X, x` | toạ độ mét (base frame) / chuẩn hoá ∈ [−1,1]³ | — |
| `F` | patch token một cửa sổ, ℝ^(B×T×VP×D) | — |

## 1. Luồng forward, pixel → keypoint

| # | phép toán | shape ra | ghi chú |
|---|---|---|---|
| 1 | frames | `(T, H, W, 3)` uint8 | frame đóng gói, có thể nhiều panel camera |
| 2 | `normalize_frames` | `(B·T, 3, H, W)` ∈ [0,1] | chia 255, permute |
| 3 | `view_crops` | V × `(B·T, 3, h_v, w_v)` | cắt panel theo `ViewLayout`; frame sai kích thước bị từ chối |
| 4 | interpolate bilinear | `(B·T, 3, 768, 768)` | mỗi crop về vuông 768 |
| 5 | `(x − μ)/σ` | như trên | μ, σ chuẩn ImageNet |
| 6 | DINOv3 ViT-L/16 | `(B·T, 1+4+2304, 1024)` | `no_grad` + autocast fp16; CLS + 4 register đứng đầu |
| 7 | bỏ 5 token đầu | `(B·T, 2304, 1024)` | lưới patch 48×48 = (768/16)² |
| 8 | avg-pool 2×2 | `(B·T, 576, 1024)` | lưới 24×24; fp16 |
| 9 | stack view | `(B·T, V, 576, 1024)` → `(B, T, V·576, 1024)` | |
| 10 | `head.sample` | `(B, T, K, 3)` mét | DDIM 10 bước × 5 mẫu, cửa sổ ≤ 64 frame |
| 11 | detector | `(T,)` mỗi detector | hình học đóng trên P |

Bước 1–9 không có tham số học. Khi huấn luyện chúng chạy một lần rồi cache ra
đĩa, vòng lặp train chỉ đọc `(T, VP, D)` fp16.

## 2. Backbone

Ký hiệu ViT là `Φ`. Với mỗi view *u*, frame *τ*:

```
F^(u)_τ = Pool_2×2( Φ( Norm( Resize_768( Crop_u(I_τ) ) ) )[5:] ) ∈ ℝ^(576×1024)
F_τ     = concat_{u=1..V} F^(u)_τ ∈ ℝ^(VP×D),  VP = 576·V
```

Pool là trung bình 2×2 trên lưới 48×48 sau khi reshape token thành ảnh đặc
trưng. Backbone đóng băng hoàn toàn (`requires_grad_(False)`, eval). Danh tính
`dinov3_vitl16@768:p2` ghi vào cache và checkpoint nên token sinh bởi cấu hình
khác không thể lẫn.

## 3. Head — kiến trúc

Head là `g_θ(x_t, t, F) → x̂₀`, ánh xạ toạ độ nhiễu về toạ độ sạch có điều
kiện trên token ảnh. Bốn khối tham số.

### 3.1 Chuẩn hoá workspace (không tham số học)

Hộp `[ℓ, h] ⊂ ℝ³` khớp một lần trước bước train đầu tiên, từ mọi target train,
lề 5 % mỗi trục:

```
ℓ = min_{ep,τ,k} X − 0.05·(max−min)
h = max_{ep,τ,k} X + 0.05·(max−min)

x = 2(X − ℓ) ⊘ (h − ℓ) − 1        X = (x + 1) ⊙ (h − ℓ)/2 + ℓ
```

Hộp lưu trong checkpoint; đọc trước khi khớp hộp là lỗi. Nửa biên độ
`(h−ℓ)/2` dùng lại trong loss.

### 3.2 Nhúng toạ độ và mức nhiễu

Fourier toạ độ, 10 octave:

```
γ(x) = [ x, sin(2⁰πx), cos(2⁰πx), …, sin(2⁹πx), cos(2⁹πx) ] ∈ ℝ⁶³
φ(x) = W₂ · SiLU(W₁ γ(x)),   W₁ ∈ ℝ^(512×63),  W₂ ∈ ℝ^(512×512)
```

63 = 3 trục × (2·10 + 1). Mười octave phủ tới 2⁹π trên [−1,1], đủ phân biệt hai
điểm cách nhau ~0.2 % biên độ workspace.

Sinusoid mức nhiễu:

```
ω_j = exp(−ln(10⁴)·j/256),  j = 0..255
e(t) = [ cos(1000·t·ω), sin(1000·t·ω) ] ∈ ℝ⁵¹²
ψ(t) = W₄ · SiLU(W₃ e(t))
```

Hệ số 1000 đưa `t ∈ (0,1]` về thang mà bảng tần số vốn thiết kế cho chỉ số bước
rời rạc.

### 3.3 Key/value từ token ảnh — tính một lần

```
kv = LN₂( W_p · LN₁(F) + pos ),          W_p ∈ ℝ^(512×1024)
pos[u,r,c] = p_view[u] + p_row[r] + p_col[c] ∈ ℝ⁵¹²   (trunc-normal 0.02)
K = W_k kv,  V = W_v kv  → reshape (B·T, 8, VP, 64)
```

**Chi tiết cài đặt:** K và V tính *một lần* cho cả cửa sổ, dùng chung cho *cả
hai* cross block, và dùng lại qua toàn bộ 10 bước DDIM × 5 mẫu. Chỉ query đổi
theo bước khử nhiễu. Đó là lý do 50 lượt lấy mẫu vẫn rẻ: phần đắt nhất — chiếu
576·V token — chạy đúng một lần.

### 3.4 Query keypoint

```
q_{τ,k} = φ(x_{t,τ,k}) + ψ(t) + e_k,   e_k ∈ ℝ⁵¹² học được, một vector mỗi keypoint
```

`ψ(t)` phát tán trên mọi `(τ, k)` — một cửa sổ dùng chung một mức nhiễu. `e_k`
là thứ duy nhất phân biệt keypoint này với keypoint kia khi `x_t` hoàn toàn
ngẫu nhiên.

### 3.5 Cross-attention lên token ảnh — 2 block

Pre-LN, không mask, các frame độc lập (N = B·T):

```
q̃ = reshape(W_q LN(q), (N, K, 8, 64))ᵀ
a  = softmax( q̃ Kᵀ / √64 ) V                    (scaled_dot_product_attention)
q  ← q + W_o · reshape(a, (N, K, 512))
q  ← q + W₆ · Drop_0.1( GELU( W₅ LN(q) ) ),     W₅ ∈ ℝ^(2048×512), W₆ ∈ ℝ^(512×2048)
```

Sau hai block: `z = LN(q) ∈ ℝ^(B×T×K×512)`. Chi phí attention là `K×VP` mỗi
frame — với K ≤ 22 và VP = 576 thì rẻ; phần nặng nằm ở bước chiếu kv.

### 3.6 Bộ mã hoá thời gian — 4 lớp, theo track

```
z_trk = reshape( permute(z, B K T d), (B·K, T, 512) )
z_trk ← TransformerEncoder_4lớp( z_trk + p_time[:T] )
z     ← z + permute⁻¹(z_trk)
```

Lớp encoder: `norm_first=True`, self-attention 8 head hai chiều (không mask
nhân quả), FF 2048, GELU, dropout 0.1. Bảng vị trí `p_time ∈ ℝ^(64×512)` — clip
dài hơn 64 frame bị chia cửa sổ ở tầng reader, không nội suy.

Attention thời gian chạy *trong một keypoint*, không chéo keypoint. Ràng buộc
liên-keypoint (xương cứng) **không** được cài vào kiến trúc — chủ ý: nếu head
biết xương phải cứng thì detector rigidity không còn đo được gì.

### 3.7 Đầu ra residual

```
x̂₀ = x_t + W_out · LN(z),   W_out ∈ ℝ^(3×512) khởi tạo 0 (cả weight lẫn bias)
```

Khởi tạo 0 nên ở bước đầu huấn luyện `g_θ` là ánh xạ đồng nhất `x_t ↦ x_t`:
mạng bắt đầu từ "không biết gì thì trả lại đầu vào", gradient đầu tiên đi qua
nhánh residual chứ không phá phần đã học.

### 3.8 Số tham số

| khối | tham số | gồm |
|---|---|---|
| temporal encoder | 12.64 M | 4 × (self-attn 512 + FF 2048) + bảng vị trí 64×512 |
| decoder cross-attn | 6.34 M | proj 1024→512, k/v proj, 2 block, pos view/row/col, query keypoint |
| time_mlp | 0.53 M | 2 × Linear 512×512 |
| coord_embed | 0.30 M | Linear 63→512, Linear 512→512 |
| x0_head + LN | < 0.01 M | Linear 512→3 |
| **tổng head** | **19.81 M** | gần như không đổi theo K |
| backbone (đóng băng) | ~300 M | không cập nhật, không lưu trong checkpoint |

## 4. Mô hình diffusion

### 4.1 Quá trình thuận (VP, lịch cosine liên tục)

```
ᾱ(t) = clip_[1e-5, 1] [ cos²((t+s)/(1+s)·π/2) / cos²(s/(1+s)·π/2) ],  s = 0.008
q(x_t | x₀) = 𝒩( √ᾱ(t)·x₀, (1 − ᾱ(t))·I ),   t ∈ (0, 1] liên tục
```

`t` liên tục, không rời rạc hoá — mỗi cửa sổ huấn luyện lấy một `t ~ 𝒰(1e-4, 1)`
duy nhất, dùng chung cho mọi frame và mọi keypoint trong cửa sổ đó. Nhiễu `ε`
độc lập từng phần tử.

### 4.2 Tham số hoá x₀ thay vì ε

Mạng dự đoán trực tiếp toạ độ sạch. Với keypoint, `x₀` có đơn vị vật lý (mét
sau khi giải chuẩn hoá) và có ràng buộc miền — nằm trong hộp workspace — nên
**clip được vào [−1,1]** ở mọi bước lấy mẫu. Tham số hoá theo `ε` không cho
phép ràng buộc đó trực tiếp. Ở `t` gần 1, dự đoán `ε` gần như vô định (tín hiệu
đã mất), trong khi dự đoán `x₀` vẫn là "đoán tư thế hợp lý nhất theo ảnh" —
đúng thứ ta muốn học.

### 4.3 Hàm mất mát

**Chỉ số.** `b ∈ {1..B}` cửa sổ, `τ ∈ {1..T}` frame, `k ∈ {1..K}` keypoint,
`c ∈ {1,2,3}` trục.

**Đại lượng vào.** Target sạch đã chuẩn hoá, một mức nhiễu mỗi cửa sổ, nhiễu
độc lập từng phần tử:

```
x₀[b,τ,k,c] = enc( X^FK[b,τ,k,c] )                     ∈ [−1,1]
t_b         ~ 𝒰(1e-4, 1)                               một giá trị mỗi cửa sổ
ε[b,τ,k,c]  ~ 𝒩(0,1)                                   i.i.d.
x_t[b,τ,k,c]= √ᾱ(t_b)·x₀[b,τ,k,c] + √(1−ᾱ(t_b))·ε[b,τ,k,c]
x̂₀          = g_θ(x_t, t, F)
δ           = x̂₀ − x₀
```

**Huber theo phần tử.** Với ngưỡng chuyển `β'`:

```
        ⎧ δ² / (2β')        nếu |δ| < β'
H_β'(δ) ⎨
        ⎩ |δ| − β'/2        ngược lại
```

Liên tục và khả vi tại `|δ| = β'`; đạo hàm `H'_β'(δ) = δ/β'` khi trong ngưỡng,
`sign(δ)` khi ngoài — **chặn bởi 1**, nên một frame đọc trượt đóng góp gradient
có giới hạn thay vì chi phối cả batch như MSE.

**Rút gọn — theo đúng thứ tự trong code.** Trung bình trên `(k, c)` trước để ra
một số mỗi frame, rồi trung bình có trọng số mask trên *toàn bộ* các cặp
`(b, τ)`:

```
L[b,τ] = (1 / 3K) · Σ_{k,c} H_β'( δ[b,τ,k,c] )

           Σ_{b,τ} m[b,τ] · L[b,τ]
ℒ(θ) =  ──────────────────────────────
         max( Σ_{b,τ} m[b,τ] , 1e-8 )
```

`m[b,τ] ∈ {0,1}` đánh dấu frame thật. Mẫu số là **tổng mask trên cả batch**,
không phải trung bình theo từng cửa sổ rồi mới trung bình batch — nên cửa sổ
nhiều frame hợp lệ hơn có trọng số lớn hơn theo tỉ lệ. Thực tế gần như mọi cửa
sổ đủ 16 frame nên hai cách trùng nhau.

**Thang của β'.** `β = 0.05 m` là ngưỡng vật lý, nhưng `δ` sống trong không
gian chuẩn hoá, nên chia cho nửa biên độ trung bình của hộp workspace:

```
s  = (1/3) Σ_c (h_c − ℓ_c) / 2          # mét, một nửa cạnh trung bình
β' = β / s                              # không thứ nguyên
```

Không chia thì "5 cm" mang nghĩa khác nhau giữa robot có workspace 1 m và robot
có 3 m — cùng một β sẽ là vùng bậc hai rộng ở robot này và hẹp ở robot kia.

**Kỳ vọng đầy đủ.** Cái tối ưu hoá thật sự là:

```
ℒ(θ) = E_{(X,F) ~ 𝒟}  E_{t ~ 𝒰(1e-4,1)}  E_{ε ~ 𝒩(0,I)}  [ ℒ(θ; x₀, t, ε) ]
```

Ước lượng Monte-Carlo bằng **một** cặp `(t, ε)` cho mỗi cửa sổ ở mỗi bước; batch
32 cửa sổ cho 32 mức nhiễu độc lập mỗi bước.

**Không có trọng số theo t** — mọi mức nhiễu đóng góp như nhau (dạng `L_simple`).
**Không có loss phụ** — không phạt độ dài xương, không phạt mượt thời gian. Thêm
ràng buộc xương vào loss sẽ dạy head vẽ ra robot cứng kể cả khi ảnh cho thấy
ngược lại, tức là làm hỏng đúng phép đo mà bench dựa vào.

Cài đặt: `DiffusionKeypointHead.training_loss` (`heads/diffusion.py`) dựng
`x_t`, gọi `g_θ`, rồi rút gọn qua `masked_smooth_l1` (`heads/blocks.py`). Đây là
loss duy nhất trong đường huấn luyện.

`kinescore train --loss mse` thay `H_β'` bằng `δ²`, cùng mask và cùng mẫu số;
`β` khi đó không dùng đến. Mọi số val_mm trong repo đo bằng `smooth_l1`.

### 4.4 Mục tiêu giám sát đến từ đâu

Không có nhãn keypoint thủ công. Target dựng từ joint teleop đã log bằng forward
kinematics của URDF:

```
X^FK_τ = FK_robot( q_τ, a_τ ) ∈ ℝ^(K×3)
```

`q` là vector khớp, `a` là độ mở gripper chuẩn hoá về [0,1]: kênh vốn đã nằm
trong [0,1] giữ nguyên, kênh vượt ra thì chia cho giá trị lớn nhất của episode.
Robot có ngón tay chuyển động ngoài `q` mà bỏ `a` sẽ ghim keypoint ngón ở tư thế
đóng suốt clip — head học vị trí ngón mâu thuẫn với pixel mỗi khi tay mở.

A1X là ngoại lệ: trạng thái log là pose đầu công tác `(x, y, z, roll, pitch,
yaw, độ mở)`, không phải góc khớp — FK của nó dựng K = 4 điểm từ pose đó.

## 5. Vòng lặp huấn luyện

### 5.1 Lấy mẫu cửa sổ

Episode dài `T_ep` frame cho một cửa sổ bằng cách chọn điểm bắt đầu
`s ~ 𝒰{0, …, T_ep−16}`; episode ngắn hơn 16 frame lấy trọn rồi đệm. Token đọc
từ cache trên đĩa qua 16 luồng; có thể giữ một bể cửa sổ thường trú trong RAM và
thay `buffer_refresh = 4` cửa sổ mỗi bước.

### 5.2 Tối ưu

| mục | giá trị |
|---|---|
| optimizer | AdamW, weight_decay 1e-4, β mặc định (0.9, 0.999) |
| batch | 32 cửa sổ × 16 frame |
| bước | 6 000 |
| learning rate | 1e-3, cosine annealing tới bước 6 000 |
| đổi nhịp | tại bước 1 500: `lr := 5e-4` và **khởi động lại** cosine với `T_max = 4500` |
| gradient | không clip, không tích luỹ, fp32 |
| EMA / guidance | không dùng |
| seed | 0 — gieo cả khởi tạo optimizer lẫn bộ lấy mẫu cửa sổ |

Lịch lr là hai chặng cosine nối nhau, không phải một cosine liên tục.

### 5.3 Đánh giá và chọn checkpoint

Mỗi 500 bước, chấm toàn bộ split val bằng **đúng đường lấy mẫu lúc suy luận**
(DDIM 10 bước, 5 mẫu, cửa sổ 64 frame):

```
val_mm = 1000 · sqrt( mean_{ep,τ,k} ‖ X̂_{τ,k} − X^FK_{τ,k} ‖² )
```

RMSE trên mọi keypoint mọi frame, milimét. Giữ state có `val_mm` thấp nhất;
checkpoint cuối là state đó, kèm `train_mm`, `val_mm`, `best_step`. Vì lấy mẫu
có ngẫu nhiên, `val_mm` mang nhiễu lấy mẫu — hai lần chấm cùng một state không
cho số y hệt.

### 5.4 Split

Chia train/val **theo scene**: khoá scene suy ra từ id episode bằng cách bỏ chỉ
số đuôi, mọi episode cùng scene đi cùng một phía, mục tiêu 15 % theo *số
episode*. Val vì thế đo khái quát sang cảnh chưa thấy, không phải sang frame kề.

## 6. Suy luận

### 6.1 DDIM tất định

Lưới `t₀ = 1 > t₁ > … > t_S = 1e-4`, S = 10, chia đều tuyến tính. Khởi tạo
`x ~ 𝒩(0, I)`. Với `i = 0 … S−1`:

```
x̂₀ = clip_[−1,1] g_θ(x, t_i, F)
ε̂  = ( x − √ᾱ_i · x̂₀ ) / max( √(1−ᾱ_i), 1e-4 )
x  ← √ᾱ_{i+1} · x̂₀ + √(1−ᾱ_{i+1}) · ε̂          (η = 0: không bơm nhiễu mới)
```

Với η = 0 quỹ đạo tất định khi đã cố định nhiễu khởi tạo — toàn bộ ngẫu nhiên
nằm ở `x` ban đầu.

**Chi tiết dễ bỏ sót:** giá trị trả về là **x̂₀ của vòng lặp cuối**, không phải
`x` sau bước cập nhật cuối. Bước cập nhật thứ S vì thế bị bỏ; nó chỉ tồn tại để
đưa `x` tới `t_S` nếu còn vòng nữa.

### 6.2 Trung bình nhiều mẫu

```
X̂ = decode( (1/N) Σ_{n=1..N} x̂₀^(n) ),   N = 5
```

Năm quỹ đạo độc lập từ năm nhiễu khởi tạo khác nhau, dùng chung K/V. Trung bình
lấy trên toạ độ chuẩn hoá rồi mới giải chuẩn hoá — hai thứ tự cho cùng kết quả
vì phép giải chuẩn hoá là affine.

Trung bình mẫu là cách xử lý mơ hồ: khi ảnh không quyết định được vị trí (che
khuất, nhoè), năm mẫu tản ra và trung bình lùi về kỳ vọng hậu nghiệm; khi ảnh
rõ, năm mẫu trùng nhau và trung bình không mất gì. Hồi quy điểm không tách được
hai trường hợp này — nó luôn trả kỳ vọng, kể cả khi không nên.

### 6.3 Cắt cửa sổ khi đọc clip dài

Reader mã hoá frame theo lô `frame_chunk` (chỉ ảnh hưởng bộ nhớ, không đổi số),
rồi chạy head theo cửa sổ **không chồng** 64 frame và nối kết quả. Ranh giới cửa
sổ cắt attention thời gian: frame 63 và 64 không thấy nhau. Segment chấm dài 16
frame nên ranh giới rơi đúng biên segment — một segment không bao giờ bị cắt đôi.

## 7. Từ keypoint xuống chỉ số

Với `P ∈ ℝ^(T×K×3)` mét và bước thời gian `Δt` giây.

**rigidity — mm**

```
r_τ = max_{(a,b) ∈ ℬ} | ‖P_{τ,a} − P_{τ,b}‖ − L_ab | · 1000
```

`ℬ` là tập cặp xương thật sự cứng của robot, `L_ab` là chiều dài nghỉ từ URDF.
Loại khỏi `ℬ`: xương suy biến (dài ~0) và xương bắc qua khớp quay hoặc qua link
do gripper truyền động — những xương này có chiều dài phụ thuộc tư thế, coi là
cứng sẽ chế ra vi phạm từ chuyển động bình thường.

**jerk — mm/s³**

```
j_τ = max_k ‖ P_{τ,k} − 3P_{τ−1,k} + 3P_{τ−2,k} − P_{τ−3,k} ‖ · 1000 / Δt³
```

Ba frame đầu bằng 0. Chia `Δt³` nên ngưỡng calibrate ở một frame-rate dùng được
ở frame-rate khác; nếu để sai phân thuần theo frame thì generator lấy mẫu thưa
nhất sẽ bị xếp là giật nhất.

**Gộp segment và ngưỡng**

```
v_i^rig  = median( r_{16i … 16i+15} )
v_i^jerk = max(    j_{16i … 16i+15} )

θ_d      = p95 { v_i^d : segment i của clip real }
ratio_i  = v_i^d / θ_d
```

Ngưỡng lấy trên *đúng thống kê* mà verdict dùng, nên theo định nghĩa 5 % segment
real vượt ngưỡng — con số này là phép kiểm tra calibrate, không phải kết quả.

## 8. Ngân sách tính toán

| đại lượng | giá trị | ghi chú |
|---|---|---|
| token / frame | 576 · V | V = 1 với ba reader hiện dùng |
| tensor token, batch train | 1.21 GB | 32 × 16 × 576 × 1024 fp32 |
| cross-attn / frame | K × 576 | ≤ 22 × 576 — rẻ; chi phí thật là chiếu 576 token |
| temporal / cửa sổ | B·K chuỗi dài T | 32 × 22 = 704 chuỗi 16 bước |
| lượt gọi `g_θ` khi đọc | 50 | 10 bước × 5 mẫu, K/V dùng lại toàn bộ |
| tham số cập nhật | 19.81 M | backbone ~300 M đóng băng |

## 9. Lựa chọn thiết kế và lý do

| lựa chọn | lý do |
|---|---|
| dự đoán keypoint, không dự đoán góc khớp | qua FK, chiều dài xương cố định theo cấu tạo → rigidity ≡ 0, mất detector chính |
| diffusion thay vì hồi quy | hồi quy trả kỳ vọng khi ảnh mơ hồ — đúng những frame cần đọc chính xác nhất |
| tham số hoá x₀, không phải ε | x₀ có miền vật lý biết trước → clip được vào hộp workspace mỗi bước |
| `W_out` khởi tạo 0 | bước đầu là ánh xạ đồng nhất, gradient không phá phần đã học |
| K/V tính một lần | 50 lượt khử nhiễu chỉ trả giá cho nhánh query |
| attention thời gian theo track, không chéo keypoint | không cài ràng buộc liên-keypoint vào mạng — đó là thứ detector phải đo |
| không loss phụ về xương / độ mượt | cùng lý do: loss như thế dạy head vẽ robot cứng bất chấp pixel |
| backbone đóng băng | token cache một lần; danh tính reader chỉ phụ thuộc head; so sánh giữa cell không lẫn biến thể backbone |
| split theo scene | `val_mm` đo khái quát sang cảnh mới, không phải nội suy frame kề |
