# Máy học trong Thị giác máy tính - Bài thực hành 1

-------------------- Thông tin sinh viên ------------------

Mã lớp học: CS332.I11.KHTN

Lớp: KHTN2014

Họ tên: Hoàng Hữu Tín

MSSV: 14520956

-------------------------- Tiến độ -------------------------

Tiến độ hiện tại:
  + Bài 1: Hoàn thành
  + Bài 2: Hoàn thành
           
           K-means, Spectral, DBSCAN, Agglomerative và visualize kết quả.
  
           Performance evaluation theo: Adjusted Rand index (ARI), Mutual Information based scores (AMI),
           Homogeneity, completeness and V-measure 
          
           --> Done
  + Bài 3: Hoàn thành
  
           File "Bai_tap_3_visualize.py" 
           
                Dataset min_face_per_person = 60
                
                Rút trích LBP histogram, chạy Clustering K-means, Spectral, DBSCAN, Agglomerative 
                và visualize kết quả.
                
                Performance evaluation theo: ARI, AMI, Homogeneity, completeness and V-measure
                
                
            File "Bai_tap_3_all.py" 
            
                Dataset toàn bộ 13233 ảnh trong LFW.
                Rút trích LBP histogram, chạy Clustering K-means + Performance evaluation.
                Thời gian chạy quá lâu nên không thực hiện các phương pháp còn lại
           
           Dữ liệu LBP features và model được lưu lại trong các file .data, .model
           --> Done
           
  + Bài 4: Hoàn thành
  
        Dataset: LFW raw (gốc chưa alignment: http://vis-www.cs.umass.edu/lfw/lfw.tgz)
        Feature: Facenet Embeddings (128 dimensional space vector)
            Paper: https://arxiv.org/pdf/1503.03832.pdf
            
        Tiến trình thực hiện:
              1. Alignment LFW raw theo Muti-task CNN: 
                    https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
              2. Training facenet theo aligned LFW
              3. Sử dụng facenet model đã train tính các embeddings cho các ảnh trong aligned LFW
              4. Dùng facenet embeddings như features để Clustering.
        
        
Báo cáo: Đang hoàn thiện ...
