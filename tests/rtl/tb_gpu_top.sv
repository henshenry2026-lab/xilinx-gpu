`timescale 1ns/1ps
module tb_gpu_top;
    reg clk_pix = 0;
    always #(500000000.0 / 25175000.0) clk_pix = ~clk_pix;
    reg rst = 1, uart_rx_pin = 1;
    wire [7:0] rgb;
    wire hsync, vsync, de;
    integer ref_x = 0, ref_y = 0, frame = 0;
    integer pixels = 0, hsync_clocks = 0, vsync_clocks = 0;
    integer active_x = 0, active_y = 0, active_w = 0, active_h = 0;
    integer pending_x = 0, pending_y = 0, pending_w = 0, pending_h = 0;
    reg [7:0] active_color = 0, pending_color = 0, expected;
    reg expected_de, expected_hsync, expected_vsync;
    reg [7:0] stimulus [0:25];
    integer image_file, i;

    gpu_top dut (.clk_pix(clk_pix), .rst(rst), .uart_rx_pin(uart_rx_pin),
                 .rgb(rgb), .hsync(hsync), .vsync(vsync), .de(de));

    task transmit_packet(input integer base);
        integer b, bit_index;
        begin
            for (b = 0; b < 13; b = b + 1) begin
                @(negedge clk_pix); uart_rx_pin = 0;
                repeat (219) @(negedge clk_pix);
                for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
                    uart_rx_pin = stimulus[base+b][bit_index];
                    repeat (219) @(negedge clk_pix);
                end
                uart_rx_pin = 1;
                repeat (219) @(negedge clk_pix);
            end
            repeat (10) @(negedge clk_pix);
        end
    endtask

    // Black-box reference: derives every pixel and sync from the published timing.
    always @(posedge clk_pix) begin
        if (rst) begin
            ref_x = 0; ref_y = 0; frame = 0;
            pixels = 0; hsync_clocks = 0; vsync_clocks = 0;
            active_w = 0; active_h = 0;
        end else begin
            expected_de = ref_x < 640 && ref_y < 480;
            expected_hsync = !(ref_x >= 656 && ref_x < 752);
            expected_vsync = !(ref_y >= 490 && ref_y < 492);
            expected = 0;
            if (expected_de && ref_x >= active_x && ref_x < active_x+active_w &&
                ref_y >= active_y && ref_y < active_y+active_h) expected = active_color;
            #1;
            if (de !== expected_de || hsync !== expected_hsync || vsync !== expected_vsync || rgb !== expected)
                $fatal(1, "frame %0d pixel (%0d,%0d): rgb=%h/%h de=%b/%b hs=%b/%b vs=%b/%b",
                    frame, ref_x, ref_y, rgb, expected, de, expected_de,
                    hsync, expected_hsync, vsync, expected_vsync);
            if (de) pixels = pixels + 1;
            if (!hsync) hsync_clocks = hsync_clocks + 1;
            if (!vsync) vsync_clocks = vsync_clocks + 1;
            if (frame == 2 && de)
                $fwrite(image_file, "%c%c%c", ((rgb >> 5) & 7) * 255 / 7,
                    ((rgb >> 2) & 7) * 255 / 7, (rgb & 3) * 255 / 3);
            if (ref_x == 0 && ref_y == 480) begin
                active_x = pending_x; active_y = pending_y;
                active_w = pending_w; active_h = pending_h; active_color = pending_color;
            end
            if (ref_x == 799) begin
                ref_x = 0;
                if (ref_y == 524) begin
                    if (pixels != 307200 || hsync_clocks != 50400 || vsync_clocks != 1600)
                        $fatal(1, "incorrect full-frame totals");
                    $display("frame %0d: 307200 active pixels, 50400 HS clocks, 1600 VS clocks", frame);
                    ref_y = 0; frame = frame + 1;
                    pixels = 0; hsync_clocks = 0; vsync_clocks = 0;
                    if (frame == 3) begin
                        $fclose(image_file);
                        $display("PASS tb_gpu_top: 3 complete frames, UART control, no tearing, demo.ppm");
                        $finish;
                    end
                end else ref_y = ref_y + 1;
            end else ref_x = ref_x + 1;
        end
    end

    initial begin
        $readmemh("build/uart_packets.mem", stimulus);
        image_file = $fopen("build/demo.ppm", "wb");
        if (!image_file) $fatal(1, "cannot create preview image");
        $fwrite(image_file, "P6\n640 480\n255\n");
        repeat (5) @(negedge clk_pix); rst = 0;
        transmit_packet(0);
        pending_x = 240; pending_y = 140; pending_w = 160; pending_h = 80; pending_color = 8'h1c;
        // Update during a visible frame. The old rectangle must persist to vblank.
        wait (frame == 1 && ref_y == 200 && ref_x == 100);
        transmit_packet(13);
        pending_x = 160; pending_y = 120; pending_w = 320; pending_h = 240; pending_color = 8'h1f;
    end
    initial begin #100000000; $fatal(1, "integration test timeout"); end
endmodule
