# Báo cáo Thực hành Lab 16 — Cloud AI Environment Setup

1. Tôi dùng AWS, region us-east-1, instance type t3.micro, source commit 55539f67d7c78b43afe334a2ec3271c4bfdbbe2d.
2. Dataset Credit Card Fraud Detection có 284.807 dòng, chia train/test theo tỉ lệ 80/20 (stratified split), random seed 42.
3. Load dữ liệu mất 1.6885 giây; training mất 1.4519 giây; best iteration là 1.
4. AUC 0.9517, Accuracy 0.9988, F1 0.7203, Precision 0.6159, Recall 0.8673 trên tập test.
5. Latency 1 dòng 1.4896 ms; throughput batch 1.000 dòng 652708.37 dòng/giây; cách đo: trung bình 100 lần predict 1 dòng và đo thời gian batch 1.000 dòng.
6. CPU/RAM/Network tôi quan sát qua top, free -h và df -h: CPU 2 vCPU ổn định, RAM sử dụng 209Mi / 914Mi, disk sử dụng 2.9G / 29G (10%); ảnh đính kèm trong bài nộp.
7. Billing tại AWS Management Console ghi nhận total estimated amount used là $0.34 (chủ yếu từ NAT Gateway, EC2 và Application Load Balancer trong thời gian triển khai bài lab).
8. Tôi đã tải kết quả về máy và hoàn tất dọn dẹp hạ tầng bằng lệnh `terraform destroy`; bằng chứng dọn dẹp hiển thị Destroy complete.
