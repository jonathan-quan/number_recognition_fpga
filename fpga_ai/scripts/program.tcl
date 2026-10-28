# Programs SRAM only. Original project and configuration flash are untouched.
set root [file normalize [file join [file dirname [info script]] ..]]
set bitfile [file join $root build digit_ai_top.bit]
if {![file isfile $bitfile]} {error "Build the AI bitstream first: $bitfile"}
open_hw_manager
connect_hw_server -url localhost:3121
set targets [get_hw_targets -quiet]
if {[llength $targets]!=1} {error "Expected exactly one connected JTAG target; found $targets"}
current_hw_target [lindex $targets 0]
open_hw_target
set devices [get_hw_devices -quiet]
if {[llength $devices]!=1} {error "Expected one Basys 3 device; found $devices"}
set dev [lindex $devices 0]
if {[get_property PART $dev] ne "xc7a35t"} {error "Not an Artix-7 35T: [get_property PART $dev]"}
set_property PROGRAM.FILE $bitfile $dev
program_hw_devices $dev
refresh_hw_device $dev
puts "AI_PROGRAMMED: $dev $bitfile"
close_hw_target
disconnect_hw_server
close_hw_manager
