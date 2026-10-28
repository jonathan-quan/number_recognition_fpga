# Isolated project: never opens or changes C:/Users/ndaqu/fpga/fpga.xpr.
set root [file normalize [file join [file dirname [info script]] ..]]
set python [file join $root .. .venv Scripts python.exe]
if {![file isfile $python]} {error "Install the Python project environment first"}
# Vivado sets PYTHONHOME/PYTHONPATH for its bundled interpreter. Ignore those
# when launching this project's Python, otherwise standard-library versions mix.
puts [exec $python -E [file join $root scripts validate_export.py]]
set out [file join $root build]
set baud 115200
if {[llength $argv]>0} {set baud [lindex $argv 0]}
if {![string is integer -strict $baud] || $baud<1200 || $baud>1000000} {error "Invalid UART baud rate"}
file mkdir $out
cd $out
create_project digit_ai [file join $out project] -part xc7a35tcpg236-1 -force
set_property target_language Verilog [current_project]
add_files [glob [file join $root rtl *.v]]
set mems [glob [file join $root .. digit_recognition export *.mem]]
add_files $mems
set_property file_type {Memory Initialization Files} [get_files *.mem]
foreach f $mems {file copy -force $f [file join $out [file tail $f]]}
add_files -fileset constrs_1 [file join $root constraints basys3_ai.xdc]
set_property top digit_ai_top [get_filesets sources_1]
set_property generic "BAUD=$baud" [get_filesets sources_1]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
open_run synth_1
report_utilization -file [file join $out synthesis_utilization.rpt]
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation/bitstream failed"}
open_run impl_1
report_utilization -file [file join $out implemented_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $out timing_summary.rpt]
report_drc -file [file join $out drc.rpt]
set worst [get_timing_paths -delay_type max -max_paths 1]
if {[llength $worst] && [get_property SLACK $worst]<0} {error "100 MHz setup timing failed"}
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $hold] && [get_property SLACK $hold]<0} {error "Hold timing failed"}
file copy -force [file join $out project digit_ai.runs impl_1 digit_ai_top.bit] [file join $out digit_ai_top.bit]
puts "AI_BUILD_PASS: [file join $out digit_ai_top.bit]"
close_project
