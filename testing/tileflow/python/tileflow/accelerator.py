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
           "get_cloud_small_40_160","get_cloud_small_40_200"]


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