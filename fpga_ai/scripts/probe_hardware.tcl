open_hw_manager
connect_hw_server -url localhost:3121
set targets [get_hw_targets -quiet]
puts "AI_HW_TARGETS: $targets"
foreach target $targets {
    current_hw_target $target
    open_hw_target
    foreach dev [get_hw_devices] {puts "AI_HW_DEVICE: $dev PART=[get_property PART $dev]"}
    close_hw_target
}
disconnect_hw_server
close_hw_manager
