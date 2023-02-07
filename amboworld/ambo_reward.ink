inkling "2.0"

using Number
const NUMBER_DISPTACH_POINTS = 25
type SimState {
    dispatches:number [NUMBER_DISPTACH_POINTS],
    location_x:number,
    location_y:number,
    time: number, 
    _gym_reward: number,
    _gym_terminal: number 
}

type ObsState {
    dispatches:number [NUMBER_DISPTACH_POINTS],
    location_x:number,
    location_y:number,
}

# Action at each step is to dispatch to one the dispatch points 
type Action {
    dispatch_point: number <0 .. NUMBER_DISPTACH_POINTS-1 step 1>
}

#siemens industrial benchmark (SIB) config
# is managed by gym versions 
# possible config and the their default values are shown below: 
# RANDOM_SEED = 42
# SIM_DURATION = 5000
# NUMBER_AMBULANCES = 3
# NUMBER_INCIDENT_POINTS = 1
# INCIDENT_RADIUS = 2
# NUMBER_DISPTACH_POINTS = 25
# AMBOWORLD_SIZE = 50
# INCIDENT_INTERVAL = 60
# EPOCHS = 2
# AMBO_SPEED = 60
# AMBO_FREE_FROM_HOSPITAL = False

type ambo_Config {
    episode_length: Number.Int8,
    NUMBER_DISPTACH_POINTS: NUMBER_DISPTACH_POINTS
}

function Reward(ss: SimState) {
    return ss._gym_reward
}

function Terminal(ss: SimState) {
    return ss._gym_terminal
}

simulator Ambo_Simulator(action: Action, config: ambo_Config): SimState {
}

graph (input: ObsState): Action {

    concept cost_reduction(input): Action {
        curriculum {
            reward Reward
            terminal Terminal
            source Ambo_Simulator
            lesson minimize_cost {
                scenario {
                    episode_length: -1,
                }
            }
        }
    }
    output cost_reduction 
}