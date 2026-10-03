#developer : Supreeth Sundaresh Athreyas
#Institute : Arizona State University
#date : 10/02/2026
#purpose : to verify the syntax of a systemverilog file
#function : invokes the vsim os command
#file tested with a simple not gate and it worked

#code is copyright protected. please do not distribute without permission of the devloper or the institute
#version 1.0


''' 
This code generates files:
files generated 
    -> modelsim dependency files (work library)
    -> run.do file
    -> transcript (if vsim runs error-free)

'''



#we start with verifier.py, where it will take a sv file, and verify the syntax, using vsim os command
import os

sv_file_path = r" " #works as this is a local path

command = "vsim -c -do run.do"

start_new  = True #deletes old compiles

remove_old_compile = ''' 
if {[file exists work]} {
    file delete -force work
}
'''


do_file_content_main = f''' 
#creating a work library
vlib work

vlog {sv_file_path}

quit -f

'''


do_file_content_final = (remove_old_compile + do_file_content_main) if start_new else do_file_content_main



with open("run.do", "w") as f:
    f.write(do_file_content_final)

os.system(command)


''' 
for those who are dev the UI

textboxes -> sv_file_path

checkboxes -> start_new

buttons -> run

run button funct -> invokes the verifier.py file

'''

