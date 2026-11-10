`timescale 1ns/1ps
`default_nettype none

// 8-N-1 receiver. CLOCK_HZ / BAUD must be at least 8 for midpoint sampling.
module uart_rx #(
    parameter integer CLOCK_HZ = 25175000,
    parameter integer BAUD = 115200
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid,
    output reg        framing_error
);
    localparam integer BIT_CLOCKS = (CLOCK_HZ + BAUD / 2) / BAUD;
    localparam integer COUNT_BITS = BIT_CLOCKS > 1 ? $clog2(BIT_CLOCKS) : 1;
    localparam [2:0] IDLE = 0, START = 1, DATA = 2, STOP = 3, WAIT_IDLE = 4;

    (* ASYNC_REG = "TRUE" *) reg rx_meta, rx_sync;
    reg [2:0] state, bit_index;
    reg [COUNT_BITS-1:0] count;
    reg [7:0] shift;

    always @(posedge clk) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            count <= 0;
            bit_index <= 0;
            shift <= 0;
            data <= 0;
            valid <= 0;
            framing_error <= 0;
        end else begin
            valid <= 0;
            framing_error <= 0;
            case (state)
                IDLE: if (!rx_sync) begin
                    count <= (BIT_CLOCKS - 1) / 2;
                    state <= START;
                end
                START: if (count != 0) count <= count - 1'b1;
                else if (rx_sync) state <= IDLE;
                else begin
                    count <= BIT_CLOCKS - 1;
                    bit_index <= 0;
                    state <= DATA;
                end
                DATA: if (count != 0) count <= count - 1'b1;
                else begin
                    shift[bit_index] <= rx_sync;
                    count <= BIT_CLOCKS - 1;
                    if (bit_index == 7) state <= STOP;
                    else bit_index <= bit_index + 1'b1;
                end
                STOP: if (count != 0) count <= count - 1'b1;
                else if (rx_sync) begin
                    data <= shift;
                    valid <= 1;
                    state <= IDLE;
                end else begin
                    framing_error <= 1;
                    state <= WAIT_IDLE;
                end
                WAIT_IDLE: if (rx_sync) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
