$path = "C:\Users\devea\Desktop\algo-trading"
cd scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
./activate.ps1
cd ..
python main.py