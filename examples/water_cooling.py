from __future__ import annotations

from pylink import (
    Block,
    ContinuousBlock,
    PortSpec,
    SimulationConfig,
    Simulator,
    StepSnapshot,
    System,
)


class RoomTemperatureSource(Block):
    """输出房间温度的普通块。"""

    # 这个块没有输入，只有一个输出端口 `room_temp`。
    # 因为房间温度在这个示例里被视为恒定常量，所以它适合建模成普通 Block，
    # 不需要离散状态，也不需要连续状态。
    outputs = (PortSpec.output("room_temp"),)

    def __init__(self, room_temperature: float) -> None:
        # direct_feedthrough=False 的意思是：
        # 这个块的输出不依赖“当前时刻刚刚算出来的其它块输入”，
        # 它只依赖自己内部保存的常量。
        super().__init__(direct_feedthrough=False)
        self.room_temperature = room_temperature

    def output(self, ctx, inputs):
        # 这里直接输出恒定室温。
        return self.room_temperature


class TemperatureDifference(Block):
    """计算水温与室温之差。"""

    # 这个块负责实现 delta_t = water_temp - room_temp。
    # 它本身不保存状态，只做纯计算，所以仍然适合建模成普通 Block。
    inputs = (
        PortSpec.input("water_temp"),
        PortSpec.input("room_temp"),
    )
    outputs = (PortSpec.output("delta_t"),)

    def __init__(self) -> None:
        # 这个块的输出显然依赖当前输入，所以它是 direct-feedthrough 块。
        super().__init__(direct_feedthrough=True)

    def output(self, ctx, inputs):
        return inputs["water_temp"] - inputs["room_temp"]


class CoolingCup(ContinuousBlock):
    """根据牛顿冷却定律更新烧杯中水的温度。"""

    # 连续系统最关键的特征是：
    # 我们关心的量会随着时间连续变化，而不是只在某些离散时刻突然跳变。
    # 水温正是一个典型的连续状态，因此这里使用 ContinuousBlock。
    inputs = (PortSpec.input("delta_t"),)
    outputs = (PortSpec.output("water_temp"),)

    def __init__(self, initial_temperature: float, cooling_rate: float) -> None:
        # 这个块的输出是“当前连续状态本身”，而不是由输入直接代数计算得到。
        # 因此它不是 direct-feedthrough 块。
        super().__init__(direct_feedthrough=False)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def initial_continuous_state(self):
        # 连续状态就是“水的当前温度”，初始值设为 80 摄氏度。
        return self.initial_temperature

    def output(self, ctx, inputs):
        # output() 表示“在当前时刻，这个块向外界输出什么信号”。
        # 对于这个例子来说，输出就是当前水温。
        return ctx.continuous_state

    def derivative(self, ctx, inputs, state):
        # derivative() 表示“连续状态对时间的变化率是多少”。
        #
        # 牛顿冷却定律写成：
        #   dT/dt = -k * (T - T_room)
        #
        # 在本模型中，TemperatureDifference 块已经算好了：
        #   delta_t = T - T_room
        #
        # 所以这里直接返回：
        #   dT/dt = -k * delta_t
        #
        # 其中 k 是冷却快慢系数：
        # - k 越大，降温越快
        # - k 越小，降温越慢
        #
        # 这个 k 只是用户在示例里给出的物理参数，不是 pylink 内置模型。
        return -self.cooling_rate * inputs["delta_t"]


class TemperatureRecorder:
    """记录仿真过程中各时刻的水温，便于在终端打印关键采样点。"""

    def __init__(self) -> None:
        self.temperature_by_time: dict[float, float] = {}

    def on_simulation_start(self, plan, config) -> None:
        # 这个示例不需要在仿真开始时做额外处理，但保留该方法可以更直观地展示 observer 结构。
        return None

    def on_step(self, snapshot: StepSnapshot) -> None:
        # 这里直接从连续状态里读出水温。
        # 因为烧杯的连续状态就是水温本身，所以这是最直接的记录方式。
        self.temperature_by_time[round(snapshot.time, 6)] = snapshot.continuous_states["cup"]

    def on_simulation_error(self, error) -> None:
        # 示例里不吞掉异常，只保留一个空方法，让结构完整。
        return None

    def on_simulation_end(self, result) -> None:
        return None


def build_system() -> System:
    """构造“热水在房间中冷却”的系统图。"""

    # 单位约定：
    # - 温度单位：摄氏度
    # - 时间单位：分钟
    #
    # 这里设置一个比较直观的冷却速度。
    # k=0.10 表示每分钟按当前温差的一定比例冷却。
    initial_water_temperature = 80.0
    room_temperature = 20.0
    cooling_rate = 0.10

    system = System("water_cooling")
    system.add_block("room", RoomTemperatureSource(room_temperature))
    system.add_block(
        "cup",
        CoolingCup(
            initial_temperature=initial_water_temperature,
            cooling_rate=cooling_rate,
        ),
    )
    system.add_block("delta", TemperatureDifference())

    # 连线说明：
    # 1. 房间温度输入到温差计算块
    # 2. 烧杯当前水温输入到温差计算块
    # 3. 温差输入到烧杯块，用来计算 dT/dt
    system.connect("room.room_temp", "delta.room_temp")
    system.connect("cup.water_temp", "delta.water_temp")
    system.connect("delta.delta_t", "cup.delta_t")

    # 这条反馈路径不会形成非法代数环，原因是：
    # - delta 是 direct-feedthrough 块，它的输出依赖当前输入
    # - cup 不是 direct-feedthrough 块，它的输出来自自身连续状态
    #
    # 也就是说，反馈里经过了一个“有状态的连续块”，
    # 不是纯代数的即时互相依赖，因此 pylink 可以正常编译和仿真。
    return system


def print_summary(recorder: TemperatureRecorder) -> None:
    """打印几个关键时间点的温度，帮助人工检查结果趋势。"""

    sample_times = [i * 0.01 for i in range(int(30 / 0.01))]

    print("烧杯热水冷却示例")
    print("模型: dT/dt = -k * (T - T_room)")
    print("单位: 温度=摄氏度, 时间=分钟")
    print()
    print("关键时刻的水温:")

    for time_point in sample_times:
        temperature = recorder.temperature_by_time[round(time_point, 6)]
        print(f"t = {time_point:>4.1f} min -> T = {temperature:>6.2f} °C")

    ordered_times = sorted(recorder.temperature_by_time)
    ordered_temperatures = [recorder.temperature_by_time[time_point] for time_point in ordered_times]
    monotonic_cooling = all(
        next_temp <= current_temp + 1e-9
        for current_temp, next_temp in zip(ordered_temperatures, ordered_temperatures[1:])
    )

    print()
    print(f"是否单调降温: {monotonic_cooling}")
    print(f"最终温度是否仍高于室温 20°C: {ordered_temperatures[-1] >= 20.0}")


def main() -> None:
    system = build_system()
    recorder = TemperatureRecorder()

    # 从 0 分钟仿真到 30 分钟，步长为 0.5 分钟。
    # 这个步长足够细，适合教学演示，也方便打印整 5 分钟时刻的结果。
    config = SimulationConfig(start=0.0, stop=30.0, dt=0.01)

    Simulator().run(system, config, observer=recorder)
    print_summary(recorder)


if __name__ == "__main__":
    main()
