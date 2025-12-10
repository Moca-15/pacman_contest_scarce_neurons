import os

# command = 'python .\capture.py -b ..\..\..\my_team_test3.py -n 10'
# os.system('echo %cd%')
# for i in range(100) :
#     print("Training num: ", i)    
#     os.system(command)
#     os.system('echo %cd%')

# baseline:
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_10.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_09.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_08.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_07.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_06.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_05.py -n 10 -q')
#os.system('echo %cd%')

# Heuristic:
os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_10.py -r ..\\..\\..\\my_team_HeuristicAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_09.py -r ..\\..\\..\\my_team_HeuristicAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_08.py -r ..\\..\\..\\my_team_HeuristicAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_07.py -r ..\\..\\..\\my_team_HeuristicAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_06.py -r ..\\..\\..\\my_team_HeuristicAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_05.py -r ..\\..\\..\\my_team_HeuristicAgent.py -n 10 -q')
#os.system('echo %cd%')

# approx q
os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_10.py -r ..\\..\\..\\my_team_ApproxQlearningAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_09.py -r ..\\..\\..\\my_team_ApproxQlearningAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_08.py -r ..\\..\\..\\my_team_ApproxQlearningAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_07.py -r ..\\..\\..\\my_team_ApproxQlearningAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_06.py -r ..\\..\\..\\my_team_ApproxQlearningAgent.py -n 10 -q')
#os.system('echo %cd%')

os.system('python .\\capture.py -b ..\\..\\..\\my_team_test3_05.py -r ..\\..\\..\\my_team_ApproxQlearningAgent.py -n 10 -q')
#os.system('echo %cd%')