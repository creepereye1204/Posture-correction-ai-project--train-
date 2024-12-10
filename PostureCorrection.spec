# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['D:\\APPS\\Posture-correction-ai-project--train-\\test.py'],
    pathex=[],
    binaries=[],
    datas=[
                 ('random_forest_model12.pkl', '.'),  # 모델 파일 포함
                 ('D:/venv_list/mediapipe-proejct-E0xIZykN-py3.10/Lib/site-packages/mediapipe/modules/pose_landmark/pose_landmark_cpu.binarypb', '.')
             ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PostureCorrection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
