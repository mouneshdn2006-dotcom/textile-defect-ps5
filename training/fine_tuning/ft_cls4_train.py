from ultralytics import YOLO
m = YOLO('/home/claude/ft_runs/cls3/weights/best.pt')
m.train(data='/home/claude/ft_cls4', epochs=4, imgsz=224, batch=32, workers=2, device='cpu', optimizer='AdamW', lr0=0.0002,
        warmup_epochs=0, cos_lr=True, plots=False, project='/home/claude/ft_runs', name='cls4', exist_ok=True)
print('DONE_TRAIN')
