 
if {[file exists work]} {
    file delete -force work
}
 
#creating a work library
vlib work

vlog not_gate.sv

quit -f

