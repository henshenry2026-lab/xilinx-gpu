`timescale 1ns/1ps
`default_nettype none

// Bias, requantize and saturate each streamed GEMM result to signed INT8.
module npu_linear_top #(
    parameter integer LANES = 4,
    parameter integer MAX_M = 16,
    parameter integer MAX_N = 16,
    parameter integer MAX_K = 64
) (
    input wire clk, rst,
    input wire wr_valid,
    output wire wr_ready,
    input wire wr_sel,
    input wire [15:0] wr_addr,
    input wire signed [7:0] wr_data,
    input wire scale_wr_valid,
    output wire scale_wr_ready,
    input wire [15:0] scale_wr_channel,
    input wire signed [31:0] scale_wr_bias,
    input wire signed [31:0] scale_wr_multiplier,
    input wire [5:0] scale_wr_shift,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [15:0] cmd_m, cmd_n, cmd_k,
    input wire cmd_relu,
    output wire busy,
    output wire done, error,
    output wire out_valid,
    input wire out_ready,
    output wire [15:0] out_row, out_col,
    output wire signed [7:0] out_data,
    output wire out_last
);
    reg [MAX_N-1:0] scale_valid;
    reg signed [31:0] bias_table [0:MAX_N-1];
    reg signed [31:0] multiplier_table [0:MAX_N-1];
    reg [5:0] shift_table [0:MAX_N-1];
    reg relu_latched;
    reg scale_error, command_error;
    reg output_valid;
    reg [15:0] output_row, output_col;
    reg signed [7:0] output_data;
    reg output_last;
    reg done_pending, done_pulse;
    integer channel;

    wire core_wr_ready, core_cmd_ready, core_busy, core_done, core_error;
    wire core_out_valid, core_out_ready, core_out_last;
    wire [15:0] core_out_row, core_out_col;
    wire signed [31:0] core_out_data;
    wire dimensions_valid = cmd_m != 0 && cmd_n != 0 && cmd_k != 0
        && cmd_m <= MAX_M && cmd_n <= MAX_N && cmd_k <= MAX_K;
    reg channel_scales_valid;
    wire command_fire = cmd_valid && cmd_ready;
    wire core_cmd_valid = command_fire && dimensions_valid && channel_scales_valid;
    wire output_slot_ready = !output_valid || out_ready;
    wire write_window = cmd_ready && !cmd_valid;
    wire scale_write_bad = scale_wr_channel >= MAX_N || scale_wr_multiplier <= 0
        || scale_wr_shift > 31;
    wire signed [32:0] biased_sum =
        {{1{core_out_data[31]}}, core_out_data}
        + {{1{bias_table[core_out_col][31]}}, bias_table[core_out_col]};
    wire signed [31:0] selected_multiplier = multiplier_table[core_out_col];
    // INT32+INT32 spans [-2^32, 2^32-2]; a positive multiplier <2^31 fits in INT64.
    wire signed [63:0] product = biased_sum * selected_multiplier;
    wire [63:0] magnitude = product[63] ? (~product + 64'd1) : product;
    wire [5:0] selected_shift = shift_table[core_out_col];
    wire [63:0] rounding_offset = selected_shift == 0 ? 64'd0
        : (64'd1 << (selected_shift - 1'b1));
    wire [63:0] rounded_magnitude = (magnitude + rounding_offset) >> selected_shift;
    wire signed [63:0] scaled_value = product[63]
        ? -$signed(rounded_magnitude) : $signed(rounded_magnitude);
    wire signed [63:0] activated_value = relu_latched && scaled_value < 0
        ? 64'sd0 : scaled_value;
    wire signed [7:0] quantized_value = activated_value > 64'sd127 ? 8'sd127
        : activated_value < -64'sd128 ? -8'sd128 : activated_value[7:0];

    always @* begin
        channel_scales_valid = 1'b1;
        for (channel = 0; channel < MAX_N; channel = channel + 1)
            if (channel < cmd_n && !scale_valid[channel]) channel_scales_valid = 1'b0;
    end

    // Keep the single-job window closed while capturing the core completion pulse.
    assign cmd_ready = !rst && core_cmd_ready && !core_done && !output_valid && !done_pending;
    assign wr_ready = !rst && write_window && core_wr_ready;
    assign scale_wr_ready = !rst && write_window;
    assign busy = core_busy || core_done || output_valid || done_pending;
    assign done = done_pulse;
    assign error = core_error || scale_error || command_error;
    assign out_valid = !rst && output_valid;
    assign out_row = output_row;
    assign out_col = output_col;
    assign out_data = output_data;
    assign out_last = output_last;
    assign core_out_ready = output_slot_ready;

    npu_top #(.LANES(LANES), .MAX_M(MAX_M), .MAX_N(MAX_N), .MAX_K(MAX_K)) gemm (
        .clk(clk), .rst(rst),
        .wr_valid(wr_valid && wr_ready), .wr_ready(core_wr_ready), .wr_sel(wr_sel),
        .wr_addr(wr_addr), .wr_data(wr_data),
        .cmd_valid(core_cmd_valid), .cmd_ready(core_cmd_ready),
        .cmd_m(cmd_m), .cmd_n(cmd_n), .cmd_k(cmd_k),
        .busy(core_busy), .done(core_done), .error(core_error),
        .out_valid(core_out_valid), .out_ready(core_out_ready),
        .out_row(core_out_row), .out_col(core_out_col),
        .out_data(core_out_data), .out_last(core_out_last)
    );

    always @(posedge clk) begin
        if (rst) begin
            scale_valid <= 0;
            relu_latched <= 0;
            scale_error <= 0; command_error <= 0;
            output_valid <= 0; output_row <= 0; output_col <= 0;
            output_data <= 0; output_last <= 0;
            done_pending <= 0; done_pulse <= 0;
        end else begin
            scale_error <= 0; command_error <= 0; done_pulse <= 0;
            if (scale_wr_valid && scale_wr_ready) begin
                if (scale_write_bad) scale_error <= 1;
                else begin
                    scale_valid[scale_wr_channel] <= 1;
                    bias_table[scale_wr_channel] <= scale_wr_bias;
                    multiplier_table[scale_wr_channel] <= scale_wr_multiplier;
                    shift_table[scale_wr_channel] <= scale_wr_shift;
                end
            end
            if (command_fire) begin
                if (!dimensions_valid || !channel_scales_valid) command_error <= 1;
                else relu_latched <= cmd_relu;
            end

            if (core_out_ready) begin
                output_valid <= core_out_valid;
                if (core_out_valid) begin
                    output_row <= core_out_row;
                    output_col <= core_out_col;
                    output_data <= quantized_value;
                    output_last <= core_out_last;
                end
            end

            if (core_done) done_pending <= 1;
            if (done_pending && !output_valid) begin
                done_pending <= 0;
                done_pulse <= 1;
            end
        end
    end
endmodule
`default_nettype wire
