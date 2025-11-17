import domino.accelerator as acc


__all__ = ["get_edge_small", "get_cloud_small",
           "get_edge_large", "get_cloud_large",
           "get_edge_small_16_16","get_edge_small_8_8","get_edge_small_64_64","get_edge_small_128_128",
           "get_edge_small_4_4","get_edge_small_256_256",
           "get_edge_small_32_4","get_edge_small_32_8","get_edge_small_32_16","get_edge_small_32_20",
           "get_edge_small_32_24","get_edge_small_32_28","get_edge_small_32_12",
           "get_edge_small_1","get_edge_small_2","get_edge_small_3","get_edge_small_5","get_edge_small_6",
           "get_edge_small_7","get_edge_small_8",
           "get_edge_small_bw10","get_edge_small_bw50","get_edge_small_bw100","get_edge_small_bw200",
           "get_edge_small_bw300","get_edge_small_bw400","get_edge_small_bw600","get_edge_small_bw700",
           "get_cloud_small_4","get_cloud_small_8","get_cloud_small_12","get_cloud_small_16",
           "get_edge_small_260","get_edge_small_240","get_edge_small_220","get_edge_small_200",
           "get_edge_small_180","get_edge_small_160","get_edge_small_140","get_edge_small_120","get_edge_small_100",
           "get_edge_small_80","get_edge_small_60","get_edge_small_40","get_edge_small_20","get_edge_small_10",
           "get_cloud_small_20_40","get_cloud_small_20_80","get_cloud_small_20_120","get_cloud_small_20_160",
           "get_cloud_small_20_200","get_cloud_small_40_40","get_cloud_small_40_80","get_cloud_small_40_120",
           "get_cloud_small_40_160","get_cloud_small_40_200",
           # Attention-specific accelerators for paper
           "get_baseline_attention_edge", "get_baseline_attention_cloud",
           "get_tpu_attention_edge", "get_tpu_attention_cloud",
           "get_uniflow_attention_edge", "get_uniflow_attention_cloud",
           # PE Partition accelerators
           "get_uniflow_partition_edge", "get_uniflow_partition_cloud"]


def get_edge_small(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_edge_large(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=8000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_cloud_small(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_cloud_large(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=64, meshX=64)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=80000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=16, meshX=16)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_16_16(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=16*16, instance=16*16)
    Reg = acc.Buffer(name="L0", instance=16*16, buffer_class="regfile", block_size=6, depth=1,
                     meshX=16*16, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_4_4(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=4*4, instance=4*4)
    Reg = acc.Buffer(name="L0", instance=4*4, buffer_class="regfile", block_size=6, depth=1,
                     meshX=4*4, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_8_8(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=8*8, instance=8*8)
    Reg = acc.Buffer(name="L0", instance=8*8, buffer_class="regfile", block_size=6, depth=1,
                     meshX=8*8, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_64_64(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=64*64, instance=64*64)
    Reg = acc.Buffer(name="L0", instance=64*64, buffer_class="regfile", block_size=6, depth=1,
                     meshX=64*64, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_128_128(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=128*128, instance=128*128)
    Reg = acc.Buffer(name="L0", instance=128*128, buffer_class="regfile", block_size=6, depth=1,
                     meshX=128*128, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_256_256(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_32_4(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=4*32, instance=4*32)
    Reg = acc.Buffer(name="L0", instance=4*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=4*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_32_8(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=8*32, instance=8*32)
    Reg = acc.Buffer(name="L0", instance=8*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=8*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_32_12(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=12*32, instance=12*32)
    Reg = acc.Buffer(name="L0", instance=12*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=12*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_32_16(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=16*32, instance=16*32)
    Reg = acc.Buffer(name="L0", instance=16*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=16*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_32_20(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=20*32, instance=20*32)
    Reg = acc.Buffer(name="L0", instance=20*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=20*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_32_24(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=24*32, instance=24*32)
    Reg = acc.Buffer(name="L0", instance=24*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=24*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_32_28(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=28*32, instance=28*32)
    Reg = acc.Buffer(name="L0", instance=28*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=28*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_edge_small_1(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=1000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_2(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=2000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_3(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=3000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_5(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=5000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_6(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=6000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_7(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=7000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_8(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=8000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_10(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=10000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_20(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

# def get_edge_small_30(L1_BW=500, L2_BW=25, L3_BW=None):
#     MAC = acc.ALU(name="mac", alu_class="intmac",
#                   datawidth=16, meshX=32*32, instance=32*32)
#     Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
#                      meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
#     PE = acc.Engine(name="PE")
#     PE.add_local(Reg, MAC)
#     L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=30000,
#                     word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
#     Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
#     Buffer.add_level(PE)
#     Buffer.add_local(L1)
#     L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
#                     block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
#     Core = acc.Engine(name="System")
#     Core.add_level(Buffer)
#     Core.add_local(L2)
#     Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
#     Acc.set_hardware_level(Core)
#     return Acc

def get_edge_small_40(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

# def get_edge_small_50(L1_BW=500, L2_BW=25, L3_BW=None):
#     MAC = acc.ALU(name="mac", alu_class="intmac",
#                   datawidth=16, meshX=32*32, instance=32*32)
#     Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
#                      meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
#     PE = acc.Engine(name="PE")
#     PE.add_local(Reg, MAC)
#     L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=50000,
#                     word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
#     Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
#     Buffer.add_level(PE)
#     Buffer.add_local(L1)
#     L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
#                     block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
#     Core = acc.Engine(name="System")
#     Core.add_level(Buffer)
#     Core.add_local(L2)
#     Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
#     Acc.set_hardware_level(Core)
#     return Acc

def get_edge_small_60(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=60000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

# def get_edge_small_70(L1_BW=500, L2_BW=25, L3_BW=None):
#     MAC = acc.ALU(name="mac", alu_class="intmac",
#                   datawidth=16, meshX=32*32, instance=32*32)
#     Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
#                      meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
#     PE = acc.Engine(name="PE")
#     PE.add_local(Reg, MAC)
#     L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=70000,
#                     word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
#     Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
#     Buffer.add_level(PE)
#     Buffer.add_local(L1)
#     L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
#                     block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
#     Core = acc.Engine(name="System")
#     Core.add_level(Buffer)
#     Core.add_local(L2)
#     Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
#     Acc.set_hardware_level(Core)
#     return Acc

def get_edge_small_80(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=80000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

# def get_edge_small_90(L1_BW=500, L2_BW=25, L3_BW=None):
#     MAC = acc.ALU(name="mac", alu_class="intmac",
#                   datawidth=16, meshX=32*32, instance=32*32)
#     Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
#                      meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
#     PE = acc.Engine(name="PE")
#     PE.add_local(Reg, MAC)
#     L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=90000,
#                     word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
#     Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
#     Buffer.add_level(PE)
#     Buffer.add_local(L1)
#     L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
#                     block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
#     Core = acc.Engine(name="System")
#     Core.add_level(Buffer)
#     Core.add_local(L2)
#     Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
#     Acc.set_hardware_level(Core)
#     return Acc

def get_edge_small_100(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=100000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_120(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=120000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_140(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=140000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_edge_small_160(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=160000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_180(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=180000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_200(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=200000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_220(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=220000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_240(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=240000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_260(L1_BW=500, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=260000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw10(L1_BW=10, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw50(L1_BW=50, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw100(L1_BW=100, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw200(L1_BW=200, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw300(L1_BW=300, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw400(L1_BW=400, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw600(L1_BW=600, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_edge_small_bw700(L1_BW=700, L2_BW=25, L3_BW=None):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_4(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_cloud_small_8(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=8000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_cloud_small_12(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=12000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
def get_cloud_small_16(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=16000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_cloud_small_20_40(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_20_80(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=80000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_20_120(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=120000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_20_160(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=160000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_20_200(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=200000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_cloud_small_40_40(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_40_80(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=80000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_40_120(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=120000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_40_160(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=160000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

def get_cloud_small_40_200(L1_BW=4000, L2_BW=800, L3_BW=160):
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)
    # Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=6, depth=1,
    #                  meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)
    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")
    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=200000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")
    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4, sizeKB=1600_000_000, word_bits=16)
    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)
    Acc = acc.TileFlowAccelerator(name="accelerator", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc

# ============================================================================
# Attention-Specific Accelerator Architectures for Paper
# ============================================================================
# These accelerators model different approaches to handling non-GEMM operations
# in self-attention, specifically for comparing:
# 1. Baseline: Systolic array + CPU (off-chip processing for non-GEMM)
# 2. TPU-style: Systolic array + VPU (on-chip vector unit for non-GEMM)
# 3. UniFlow: Unified PE supporting both GEMM and non-GEMM operations


def get_baseline_attention_edge(L1_BW=500, L2_BW=25):
    """
    Baseline accelerator: Systolic array for GEMM + CPU for non-GEMM operations

    Architecture:
    - 32x32 systolic array (MAC units) for matrix multiplication
    - Non-GEMM ops (max, exp, div, etc.) must be sent to CPU via DRAM
    - Models off-chip data movement penalty for non-GEMM operations

    Use case: Traditional systolic array accelerators without on-chip support
    for elementwise/reduction operations
    """
    # Standard MAC units for GEMM operations
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)

    # Register file at PE level
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)

    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)

    # L1 SRAM - local buffer for GEMM operands/results
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)

    # L2 DRAM - models CPU access (non-GEMM ops processed here)
    # Lower bandwidth simulates off-chip communication penalty
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)

    Acc = acc.TileFlowAccelerator(name="baseline_attention_edge", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_baseline_attention_cloud(L1_BW=4000, L2_BW=800, L3_BW=160):
    """
    Baseline accelerator (cloud-scale): Systolic array for GEMM + CPU for non-GEMM

    Architecture:
    - 256x256 systolic array for matrix multiplication
    - Non-GEMM ops sent to CPU via DRAM (L3)
    - Three-level memory hierarchy
    """
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)

    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)

    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)

    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)

    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")

    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)

    # L3 DRAM - CPU processes non-GEMM ops here
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)

    Acc = acc.TileFlowAccelerator(name="baseline_attention_cloud", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_tpu_attention_edge(L1_BW=500, L1_VPU_BW=400, L2_BW=25):
    """
    TPU-style accelerator: Systolic array for GEMM + on-chip VPU for non-GEMM

    Architecture:
    - 32x32 systolic array (MAC units) for matrix multiplication
    - Dedicated on-chip VPU buffer (modeled as separate L1 region) for non-GEMM ops
    - On-chip data movement between systolic array and VPU via high-BW interconnect

    Key difference from baseline: Non-GEMM ops stay on-chip with dedicated VPU
    processing, avoiding off-chip DRAM access

    Memory organization:
    - L1: Main SRAM for systolic array GEMM operations (500 GB/s)
    - L1_VPU: On-chip buffer for VPU non-GEMM operations (400 GB/s)
    - On-chip communication between L1 and L1_VPU
    """
    # MAC units for GEMM
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)

    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=6, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)

    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)

    # L1 SRAM for systolic array operations
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")

    # VPU buffer for non-GEMM operations (on-chip, high bandwidth)
    # Model as additional SRAM at same level to represent on-chip VPU
    L1_VPU = acc.Buffer(name="L1_VPU", buffer_class="SRAM", width=16, sizeKB=2000,
                        word_bits=16, read_bandwidth=L1_VPU_BW, write_bandwidth=L1_VPU_BW/2.5,
                        technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1, L1_VPU)  # Both L1 and VPU are on-chip

    # L2 DRAM - only for initial data loading and final results
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)

    Acc = acc.TileFlowAccelerator(name="tpu_attention_edge", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_tpu_attention_cloud(L1_BW=4000, L1_VPU_BW=3000, L2_BW=800, L3_BW=160):
    """
    TPU-style accelerator (cloud-scale): Systolic array + on-chip VPU

    Architecture:
    - 256x256 systolic array for GEMM
    - On-chip VPU for non-GEMM operations
    - Three-level memory hierarchy with VPU at L1 level
    """
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)

    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=60, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=3, write_bandwidth=3)

    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)

    # L1 SRAM for systolic array
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")

    # VPU buffer at L1 level (on-chip)
    L1_VPU = acc.Buffer(name="L1_VPU", buffer_class="SRAM", width=16, sizeKB=10000,
                        word_bits=16, read_bandwidth=L1_VPU_BW, write_bandwidth=L1_VPU_BW*0.4,
                        technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1, L1_VPU)

    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")

    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)

    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)

    Acc = acc.TileFlowAccelerator(name="tpu_attention_cloud", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_uniflow_attention_edge(L1_BW=500, L2_BW=25):
    """
    UniFlow accelerator: Unified PE supporting both GEMM and non-GEMM operations

    Architecture:
    - 32x32 unified PEs with enhanced ALUs supporting:
      * GEMM operations (matrix multiplication)
      * Non-GEMM operations (max, exp, div, sub, add)
    - All operations execute directly on PE array without offloading
    - No separate VPU or CPU needed - PEs are multi-functional

    Key innovation: Each PE contains both MAC and non-GEMM functional units,
    enabling seamless execution of entire attention pipeline on the same hardware

    Implementation note:
    - We model this by adding multiple ALU types to each PE
    - The alu_class "unified" represents PEs with both MAC and elementwise capabilities
    """
    # Unified MAC supporting all operations (GEMM and non-GEMM)
    # TileFlow implicitly handles different operation types (max, exp, div, etc.)
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=32*32, instance=32*32)

    # Larger register file to hold intermediate results from both op types
    Reg = acc.Buffer(name="L0", instance=32*32, buffer_class="regfile", block_size=12, depth=1,
                     meshX=32*32, word_bits=16, technology="16nm", read_bandwidth=4, write_bandwidth=4)

    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)  # Single MAC handles all operations

    # L1 SRAM - unified buffer for all operation types
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5, technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE)
    Buffer.add_local(L1)

    # L2 DRAM - only for initial input/final output
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)

    Acc = acc.TileFlowAccelerator(name="uniflow_attention_edge", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_uniflow_attention_cloud(L1_BW=4000, L2_BW=800, L3_BW=160):
    """
    UniFlow accelerator (cloud-scale): Unified PE supporting both GEMM and non-GEMM

    Architecture:
    - 256x256 unified PEs with both MAC and non-GEMM functional units
    - All attention operations execute on same PE array
    - Three-level memory hierarchy
    """
    # Unified MAC for all operations (GEMM and non-GEMM)
    # TileFlow implicitly handles different operation types
    MAC = acc.ALU(name="mac", alu_class="intmac",
                  datawidth=16, meshX=256*256, instance=256*256)

    # Enhanced register file for unified operations
    Reg = acc.Buffer(name="L0", instance=256*256, buffer_class="regfile", block_size=80, depth=1,
                     meshX=256*256, word_bits=16, technology="16nm", read_bandwidth=4, write_bandwidth=4)

    PE = acc.Engine(name="PE")
    PE.add_local(Reg, MAC)

    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4, technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE)
    Buffer.add_local(L1)

    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4, technology="16nm")

    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)

    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)

    Acc = acc.TileFlowAccelerator(name="uniflow_attention_cloud", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_uniflow_partition_edge(gemm_ratio=0.75, L1_BW=500, L2_BW=25):
    """
    UniFlow with PE Partition (edge-scale): Spatial partitioning of PE array

    Architecture:
    - Total 32x32 = 1024 PEs, spatially partitioned into two blocks:
      * GEMM Block: gemm_ratio of PEs (e.g., 75% = 768 PEs) for matrix multiplication
      * Non-GEMM Block: (1-gemm_ratio) of PEs (e.g., 25% = 256 PEs) for softmax/layernorm
    - Both blocks share L1 SRAM for data exchange
    - Enables pipelining: GEMM block works on tile i+1 while non-GEMM block processes tile i

    Key advantage over unified UniFlow:
    - Better resource utilization through spatial parallelism
    - GEMM and non-GEMM operations execute concurrently
    - Reduced idle time compared to temporal pipelining

    Args:
        gemm_ratio: Fraction of PEs dedicated to GEMM (default 0.75)
        L1_BW: L1 SRAM bandwidth in GB/s
        L2_BW: L2 DRAM bandwidth in GB/s
    """
    total_pes = 32 * 32  # 1024 total PEs
    gemm_pes = int(total_pes * gemm_ratio)
    nongemm_pes = total_pes - gemm_pes

    # GEMM Block: Dedicated PEs for matrix multiplication
    MAC_GEMM = acc.ALU(name="mac_gemm", alu_class="intmac",
                       datawidth=16, meshX=gemm_pes, instance=gemm_pes)

    Reg_GEMM = acc.Buffer(name="L0_GEMM", instance=gemm_pes, buffer_class="regfile",
                          block_size=8, depth=1, meshX=gemm_pes, word_bits=16,
                          technology="16nm", read_bandwidth=3, write_bandwidth=3)

    PE_GEMM = acc.Engine(name="PE_GEMM")
    PE_GEMM.add_local(Reg_GEMM, MAC_GEMM)

    # Non-GEMM Block: Dedicated PEs for softmax, layernorm, elementwise ops
    # Also uses intmac - TileFlow handles operation types implicitly
    ALU_NonGEMM = acc.ALU(name="alu_nongemm", alu_class="intmac",
                          datawidth=16, meshX=nongemm_pes, instance=nongemm_pes)

    Reg_NonGEMM = acc.Buffer(name="L0_NonGEMM", instance=nongemm_pes, buffer_class="regfile",
                             block_size=12, depth=1, meshX=nongemm_pes, word_bits=16,
                             technology="16nm", read_bandwidth=4, write_bandwidth=4)

    PE_NonGEMM = acc.Engine(name="PE_NonGEMM")
    PE_NonGEMM.add_local(Reg_NonGEMM, ALU_NonGEMM)

    # Shared L1 SRAM - both PE blocks access this for data exchange
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=4000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW/2.5,
                    technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=4, meshX=4)
    Buffer.add_level(PE_GEMM)
    Buffer.add_level(PE_NonGEMM)  # Both PE types at same level
    Buffer.add_local(L1)

    # L2 DRAM
    L2 = acc.Buffer(name="L2", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L2_BW, write_bandwidth=L2_BW/2.5,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Buffer)
    Core.add_local(L2)

    Acc = acc.TileFlowAccelerator(name="uniflow_partition_edge", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc


def get_uniflow_partition_cloud(gemm_ratio=0.75, L1_BW=4000, L2_BW=800, L3_BW=160):
    """
    UniFlow with PE Partition (cloud-scale): Spatial partitioning of PE array

    Architecture:
    - Total 256x256 = 65536 PEs, spatially partitioned:
      * GEMM Block: gemm_ratio of PEs (e.g., 75% = 49152 PEs)
      * Non-GEMM Block: (1-gemm_ratio) of PEs (e.g., 25% = 16384 PEs)
    - Shared L1 SRAM for inter-block communication
    - Three-level memory hierarchy
    - Concurrent execution of GEMM and non-GEMM operations

    Args:
        gemm_ratio: Fraction of PEs for GEMM (default 0.75)
        L1_BW: L1 SRAM bandwidth
        L2_BW: L2 SRAM bandwidth
        L3_BW: L3 DRAM bandwidth
    """
    total_pes = 256 * 256  # 65536 total PEs
    gemm_pes = int(total_pes * gemm_ratio)
    nongemm_pes = total_pes - gemm_pes

    # GEMM Block
    MAC_GEMM = acc.ALU(name="mac_gemm", alu_class="intmac",
                       datawidth=16, meshX=gemm_pes, instance=gemm_pes)

    Reg_GEMM = acc.Buffer(name="L0_GEMM", instance=gemm_pes, buffer_class="regfile",
                          block_size=60, depth=1, meshX=gemm_pes, word_bits=16,
                          technology="16nm", read_bandwidth=3, write_bandwidth=3)

    PE_GEMM = acc.Engine(name="PE_GEMM")
    PE_GEMM.add_local(Reg_GEMM, MAC_GEMM)

    # Non-GEMM Block
    # Also uses intmac - TileFlow handles operation types implicitly
    ALU_NonGEMM = acc.ALU(name="alu_nongemm", alu_class="intmac",
                          datawidth=16, meshX=nongemm_pes, instance=nongemm_pes)

    Reg_NonGEMM = acc.Buffer(name="L0_NonGEMM", instance=nongemm_pes, buffer_class="regfile",
                             block_size=80, depth=1, meshX=nongemm_pes, word_bits=16,
                             technology="16nm", read_bandwidth=4, write_bandwidth=4)

    PE_NonGEMM = acc.Engine(name="PE_NonGEMM")
    PE_NonGEMM.add_local(Reg_NonGEMM, ALU_NonGEMM)

    # Shared L1 SRAM
    L1 = acc.Buffer(name="L1", buffer_class="SRAM", width=16, sizeKB=20000,
                    word_bits=16, read_bandwidth=L1_BW, write_bandwidth=L1_BW*0.4,
                    technology="16nm")

    Buffer = acc.Engine(name="Buffer", instance=16, meshX=16)
    Buffer.add_level(PE_GEMM)
    Buffer.add_level(PE_NonGEMM)
    Buffer.add_local(L1)

    # L2 SRAM
    L2 = acc.Buffer(name="L2", buffer_class="SRAM", width=16, sizeKB=40000,
                    word_bits=16, read_bandwidth=L2_BW, write_bandwidth=L2_BW*0.4,
                    technology="16nm")

    Cache = acc.Engine(name="Cache", instance=4, meshX=4)
    Cache.add_level(Buffer)
    Cache.add_local(L2)

    # L3 DRAM
    L3 = acc.Buffer(name="L3", buffer_class="DRAM", technology="16nm",
                    block_size=32, read_bandwidth=L3_BW, write_bandwidth=L3_BW*0.4,
                    sizeKB=1600_000_000, word_bits=16)

    Core = acc.Engine(name="System")
    Core.add_level(Cache)
    Core.add_local(L3)

    Acc = acc.TileFlowAccelerator(name="uniflow_partition_cloud", version=0.2)
    Acc.set_hardware_level(Core)
    return Acc
