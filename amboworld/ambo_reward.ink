inkling "2.0"

using Number
const number_dispatch_points = 25
type SimState {
    dispatches:number [number_dispatch_points],
    location_x:number,
    location_y:number,
    time: number, 
    _gym_reward: number,
    _gym_terminal: number 
}

type ObsState {
    dispatches:number [number_dispatch_points],
    location_x:number,
    location_y:number,
}

# Action at each step is to dispatch to one the dispatch points 
type Action {
    dispatch_point: number <0 .. number_dispatch_points-1 step 1>
}

#siemens industrial benchmark (SIB) config
# is managed by gym versions 
# possible config and the their default values are shown below: 
# max_size=50,
# number_ambulances=8,
# number_dispatch_points=25,
# number_incident_points=4,
# incident_range=0.0,
# number_hospitals=1,
# duration_incidents=1e5,
# ambo_kph=60.0,
# random_seed=42,
# incident_interval=20,
# ambo_free_from_hospital=True

type ambo_Config {
    episode_length: Number.Int8,
    max_size: Number.Int8,
    number_ambulances: Number.Int8,
    number_dispatch_points: Number.Int8,
    number_incident_points: Number.Int8,
    incident_range: number,
    number_hospitals: Number.Int8,
    duration_incidents: Number.Int32,
    ambo_kph: Number.Int8,
    random_seed: Number.Int8,
    incident_interval: Number.Int8,
}

function Reward(ss: SimState) {
    return ss._gym_reward
}

function Terminal(ss: SimState) {
    return ss._gym_terminal
}

simulator Ambo_Dispatcher(action: Action, config: ambo_Config): SimState {
}

graph (input: ObsState): Action {

    concept response_time(input): Action {
        curriculum {
            reward Reward
            terminal Terminal
            source Ambo_Dispatcher
            lesson minimize_response_time {
                scenario {
                    episode_length: -1,
                    number_ambulances: Number.Int8<3 .. 10>,
                    number_incident_points: Number.Int8<3 .. 10>
                }
            }
        }
    }
    output response_time 
}