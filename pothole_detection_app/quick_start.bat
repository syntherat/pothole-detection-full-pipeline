@echo off
REM Quick Start Script for Pothole Detection Training
REM
REM NOTE: organize_dataset.py has a hardcoded SOURCE_DIR near the top of the
REM file. Point it at your raw dataset before running this, or step 1 will
REM print an error and exit 0 -- which is why each dataset step below is
REM followed by an explicit file check rather than just `if errorlevel 1`.

echo ============================================================
echo POTHOLE DETECTION - COMPLETE TRAINING PIPELINE
echo ============================================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [1/5] Organizing Dataset (-^> data\dataset_v2)...
echo ============================================================
python scripts\organize_dataset.py
if errorlevel 1 (
    echo ERROR: Dataset organization failed!
    pause
    exit /b 1
)
if not exist "data\dataset_v2\data.yaml" (
    echo ERROR: data\dataset_v2\data.yaml was not created.
    echo organize_dataset.py exits 0 even when it finds nothing, so check that
    echo its SOURCE_DIR points at a folder containing image/label pairs.
    pause
    exit /b 1
)

echo.
echo [2/5] Merging Datasets (dataset_v2 + data\raw -^> data\dataset_v3)...
echo ============================================================
echo train_model.py reads dataset_v3, so this step is required -- without it
echo training fails with "Data config not found".
python scripts\merge_datasets.py
if errorlevel 1 (
    echo ERROR: Dataset merge failed!
    pause
    exit /b 1
)
if not exist "data\dataset_v3\data.yaml" (
    echo ERROR: data\dataset_v3\data.yaml was not created.
    echo Check that data\raw\images and data\raw\annotations exist.
    pause
    exit /b 1
)

echo.
echo [3/5] Training Model (this will take 1-3 hours)...
echo ============================================================
echo Training with: YOLO11-Small, Baseline hyperparameters
echo NOTE: on success this OVERWRITES model\best.pt. Archive it first if you
echo want to keep the current weights.
echo.
python scripts\train_model.py --model small --hyperparams baseline
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo [4/5] Evaluating Model...
echo ============================================================
REM evaluate_model.py defaults to dataset_v2; pass v3 explicitly so we evaluate
REM on the same data we just trained on.
python scripts\evaluate_model.py --data data\dataset_v3\data.yaml
if errorlevel 1 (
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
)

echo.
echo [5/5] Launching Application...
echo ============================================================
python app\main_enhanced.py

echo.
echo ============================================================
echo COMPLETE!
echo ============================================================
pause
