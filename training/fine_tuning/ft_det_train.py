from ultralytics import YOLO
m = YOLO('/home/claude/real_pt/detector_best.pt')
m.train(data='/home/claude/ft_det/data.yaml', epochs=4, imgsz=512, batch=16, workers=2, device='cpu', freeze=10,
        optimizer='AdamW', lr0=0.0005, warmup_epochs=0, val=False, plots=False, close_mosaic=1,
        project='/home/claude/ft_runs', name='det', exist_ok=True)
print('DONE_TRAIN')
