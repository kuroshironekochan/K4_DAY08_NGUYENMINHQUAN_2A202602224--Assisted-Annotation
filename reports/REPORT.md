# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Minh Quân

Công cụ gán nhãn đã dùng: CVAT

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian và có vùng đệm ở giữa thay vì chia ngẫu nhiên là do đặc thù camera giao thông đứng yên một chỗ. Một chiếc xe khi di chuyển qua khung hình sẽ xuất hiện liên tục trong vài giây (nhiều khung hình liên tiếp). Nếu chia ngẫu nhiên (random split), các khung hình cách nhau chỉ vài phần mười giây của cùng một chiếc xe sẽ rơi vào cả tập học (train/pool) và tập kiểm thử (test). Khi đó, mô hình sẽ vừa học chiếc xe đó vừa được dùng chính chiếc xe đó để chấm điểm (hiện tượng rò rỉ dữ liệu - data leakage), khiến kết quả đo lường (mAP, Precision, Recall) cao giả tạo và đẹp hơn thực tế rất nhiều. Việc chia theo thời gian và tạo vùng đệm ở giữa đảm bảo xe ở tập học đã đi hẳn ra khỏi góc quay trước khi bắt đầu tập kiểm thử, phản ánh năng lực tổng quát hóa thật sự của mô hình.

## 2. Mô hình khởi đầu lạnh (cold start)

Kết quả từ `reports/rounds_table.md` của vòng 0:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Ở lần chạy đầu tiên (cold start với YOLOv8n tiền huấn luyện trên COCO), điểm khớp khung AP50 đạt 0.771. Nhìn vào độ phủ (recall) theo kích thước xe:
- Xe nhỏ (small): chỉ đạt 0.182 (18.2%)
- Xe vừa (medium): đạt 0.547 (54.7%)
- Xe lớn (large): đạt 0.561 (56.1%)

Điều này cho thấy mô hình bỏ sót rất nhiều xe ở xa (xe nhỏ) do chi tiết mờ nhạt và ánh sáng ban đêm yếu, trong khi các xe ở gần (vừa và lớn) được phát hiện tốt hơn nhiều. Quan sát ảnh `outputs/compare_round0.jpg`, mô hình khởi đầu lạnh thường bị lệch hoặc nhận diện sai ở các xe bị che khuất một phần, nhận nhầm bóng đèn phản chiếu trên mặt đường thành xe, hoặc vẽ box chưa ôm sát viền xe. Đáng chú ý, nhãn tham chiếu dùng để chấm cũng được sinh tự động bởi mô hình chứ chưa được con người rà soát từng khung hình, do đó một số trường hợp khung lệch hoặc sai sót có thể xuất phát từ chính nhãn tham chiếu chứ không hoàn toàn do mô hình của chúng ta.

## 3. Chiến lược chọn mẫu

Thuật toán tính điểm ưu tiên cho từng ảnh theo công thức:
`score = W_U · U + W_A · A + W_D · D`
Trong đó:
- `U` (Uncertainty - chiếm trọng số 0.5): Độ bất định của mô hình, đo mức độ thiếu tự tin ở các vùng dự đoán.
- `A` (Ambiguity - chiếm trọng số 0.3): Mức độ lưỡng lự, đo số lượng các box có độ tin cậy nằm trong khoảng phân vân.
- `D` (Diversity - chiếm trọng số 0.2): Tính đa dạng thời gian, ưu tiên các ảnh phân bố đều theo trục thời gian video.
- `MIN_GAP_S` (ngưỡng khoảng cách tối thiểu, mặc định 2.0s): Do camera cố định, các ảnh cách nhau dưới 2 giây có cảnh vật và vị trí các xe gần như giống hệt nhau. `MIN_GAP_S` ngăn thuật toán chọn các ảnh sát giờ nhau, tránh lãng phí công gán nhãn vào dữ liệu dư thừa.

Minh họa từ `reports/SELECTION.md`:
- Ba ảnh được chọn trong lô 12 ảnh: `frame_0182.jpg` (hạng 1, điểm 0.9591), `frame_0099.jpg` (hạng 8, điểm 0.9063) và `frame_0107.jpg` (hạng 14, điểm 0.8876) đều có điểm bất định `U` cao và chứa từ 14 đến 18 box mơ hồ, mang lại nhiều giá trị thông tin cần người can thiệp.
- Một ảnh bị loại: `frame_0372.jpg` có hạng 6 và điểm rất cao (0.9101), nhưng bị loại vì xuất hiện ở t = 148.8s, cách `frame_0369.jpg` (hạng 2, t = 147.6s) chỉ 1.2 giây (< 2.0s).

Điểm cao chỉ phản ánh việc mô hình đang phân vân hoặc chưa chắc chắn về các dự đoán trong ảnh đó, chứ chưa thể chứng minh rằng việc con người sửa ảnh đó xong thì mô hình học lại chắc chắn sẽ giỏi hơn.

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp kết quả các vòng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 271 | 0.434 | -0.337 | 1.000 | 0.122 | 0.217 | 0.000 | 0.101 | 0.463 |

Ở vòng 1:
- Mức độ sửa nhãn gợi ý (theo `outputs/round1_diff.md`): Trong 12 ảnh với 169 box gợi ý ban đầu, sau khi người rà soát đã có tổng cộng 271 box hoàn chỉnh. Trong đó: giữ nguyên (accepted) 146 box, chỉnh sửa (edited) 9 box, xóa bỏ (deleted - False Positive của AI) 14 box, và vẽ thêm mới (added - False Negative của AI) 116 box (tỷ lệ chấp nhận đạt 86%).
- Thay đổi AP50: Giảm từ 0.771 xuống 0.434 (giảm 0.337). Tuy nhiên, độ chính xác Precision@0.25 tăng lên tuyệt đối 1.000 (không còn bất kỳ box dương tính giả FP nào), nhưng độ phủ Recall@0.25 giảm từ 0.489 xuống 0.122 do mô hình trở nên quá thận trọng (FN tăng lên 354).
- Phân nhóm xe: Nhóm xe nhỏ recall giảm về 0.000, nhóm xe vừa giảm từ 0.547 xuống 0.101, nhóm xe lớn giảm từ 0.561 xuống 0.463.

Đối chiếu ảnh `outputs/compare_round0.jpg` và `outputs/compare_round1.jpg`: Sau khi huấn luyện lại với 12 ảnh đã rà nhãn, mô hình đã triệt tiêu hoàn toàn các phán đoán sai lệch về vệt sáng/biển báo. Các xe lớn ở cự ly gần được khoanh khung rất gọn và khít. Tuy nhiên, vì mới chỉ có 12 ảnh huấn luyện nên mô hình bị co cụm (conservative), chưa đủ dữ liệu khái quát hóa cho các xe nhỏ ở xa.

Phân biệt ba quá trình:
1. Quan sát độc lập (`reports/BLIND_SCAN.md`): Quét bằng mắt thường trên `frame_0099.jpg`, dự đoán 23 xe và chỉ ra các vị trí xe bị che khuất hoặc bị cắt bởi viền ảnh mà AI dễ bỏ sót.
2. Sửa nhãn thực tế (`reports/REVIEW_LOG.csv`): Đã bổ sung các xe bị khuất và bị cắt mép ảnh (`frame_0099.jpg`, `frame_0187.jpg`, `frame_0270.jpg`), đồng thời xóa các box nhận nhầm biển báo và vệt đèn phản chiếu (`frame_0187.jpg`, `frame_0331.jpg`).
3. Kết quả sau train: Mô hình học được tính kỷ luật cao (Precision = 1.000), nhưng do số lượng mẫu huấn luyện còn ít nên chưa bao quát được toàn bộ xe nhỏ/xa. Một ca khó điển hình theo `GUIDELINE_LABEL.md` là xe bị cắt ở mép khung hình chỉ nhìn thấy một phần đuôi hoặc xe ở rất xa chỉ phát ra hai chấm đèn mờ, đòi hỏi người gán nhãn phải thống nhất quy tắc chỉ khoanh phần thân xe còn thấy trong ảnh.

## 5. Kết luận và giới hạn

So với cold start (AP50 = 0.771), vòng 1 đạt AP50 = 0.434 với đặc tính Precision tăng tuyệt đối (1.000) nhưng Recall thấp (0.122). Quyết định hợp lý là tiếp tục thực hiện thêm các vòng học chủ động tiếp theo chứ không dừng lại, vì 12 ảnh ở vòng 1 là kích thước mẫu quá nhỏ để mô hình thích nghi toàn diện. 

Hai ca còn yếu và bất định cần ưu tiên rà soát ở các vòng sau:
1. Xe ở cự ly xa chỉ còn hai đốm sáng mờ hoặc thân xe chìm trong bóng tối.
2. Xe ở mép khung hình bị cắt một phần thân xe.
Chi phí rà nhãn cho các ca này khá cao do đòi hỏi phóng to và quan sát tỉ mỉ, đồng thời cần tuân thủ nghiêm ngặt quy tắc `MIN_GAP_S` để tránh chọn các khung hình trùng lặp trong phạm vi dưới 2 giây.

Các giới hạn cần lưu ý:
- Tập kiểm thử chỉ gồm 20 ảnh và tự động bỏ qua các xe quá nhỏ (dưới 16 px).
- Nhãn tham chiếu của tập test được tạo tự động bởi mô hình chứ chưa được chuyên gia thẩm định thủ công từng box, nên các chỉ số đo đạc mang tính chất định hướng tương đối.
Nếu AP50 giảm sau một vòng huấn luyện, điều quan trọng nhất trước khi tiếp tục là mở `reports/REVIEW_LOG.csv` và `outputs/round*_diff.md` để kiểm tra lại tính nhất quán của các box đã sửa, đảm bảo người gán nhãn không áp dụng các tiêu chuẩn mâu thuẫn hoặc vô tình bỏ sót hàng loạt xe tương tự trong lô.
