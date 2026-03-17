/* Compact layout: entry at 0x5000 after mailbox region (ends 0x428C). */

ENTRY(_start)

SECTIONS
{
    . = 0x5000;

    .rodata : { *(.rodata .rodata.*) }

    . = ALIGN(4);
    .bss : { *(.bss .bss.* .sbss .sbss.*) }

    . = ALIGN(4);
    .text : { *(.text .text.*) }
    .eh_frame : { *(.eh_frame) }

    /DISCARD/ : { *(.comment) *(.riscv.attributes) }
}
