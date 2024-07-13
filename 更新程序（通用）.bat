@echo OFF
@REM normalize the relative path to a shorter absolute path.
pushd "%~dp0"
set repos_path=%CD%
popd
call :find_and_pull %repos_path%
echo. & echo Finished. & pause>nul
goto :EOF

::-------------------------------------
:: @name   find_and_pull
:: @param  %1 base directory to find .git
:: @usage  call :find_and_pull %base_dir%
::-------------------------------------
:find_and_pull

    cd %repos_path%
    if exist .git (
        echo. & echo [ START git pull FROM ] %repos_path%
        git pull
    ) else (
        call :find_and_pull %repos_path%
    )

goto :EOF