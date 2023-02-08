#!/usr/bin/env python3
# coding=utf-8

"""
MSFT Bonsai SDK3 Template for Simulator Integration using Python
Copyright 2022 Microsoft

Usage:
  For registering simulator with the Bonsai service for training:
    python simulator_integration.py   
    Then connect your registered simulator to a Brain via UI, or using the CLI: `bonsai simulator unmanaged connect -b <brain-name> -a <train-or-assess> -c  --simulator-name 
"""

from functools import partial
import json
import random
import numpy as np 
import math
import time
import os
import pathlib
from dotenv import load_dotenv, set_key
import datetime
from typing import Dict, Any, Union
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig, BonsaiClient
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorState,
    SimulatorInterface,
    SimulatorSessionResponse,
)
from sim import environment
from azure.core.exceptions import HttpResponseError
import argparse
import gym
from policies import random_policy, brain_policy, forget_memory
import pdb
import pandas as pd 



LOG_PATH = "logs"


DISPLAY_ON_SCREEN = True

# SIM PARAMETER
RANDOM_SEED = 42
SIM_DURATION = 5000
NUMBER_AMBULANCES = 3
NUMBER_INCIDENT_POINTS = 1
INCIDENT_RADIUS = 2
NUMBER_DISPTACH_POINTS = 25
AMBOWORLD_SIZE = 2*NUMBER_DISPTACH_POINTS
INCIDENT_INTERVAL = 60
EPOCHS = 2
AMBO_SPEED = 60
AMBO_FREE_FROM_HOSPITAL = False
DEFAULT_CONFIG  = {
    'random_seed' : RANDOM_SEED,
    'duration_incidents' : SIM_DURATION,
    'number_ambulances' : NUMBER_AMBULANCES,
    'number_incident_points' : NUMBER_INCIDENT_POINTS,
    'incident_interval' : INCIDENT_INTERVAL,
    'number_epochs' : EPOCHS,
    'number_dispatch_points' : NUMBER_DISPTACH_POINTS,
    'incident_range' : INCIDENT_RADIUS,
    'max_size': AMBOWORLD_SIZE,
    'ambo_kph': AMBO_SPEED,
    'ambo_free_from_hospital': AMBO_FREE_FROM_HOSPITAL    
}

def ensure_log_dir(log_full_path):
    """
    Ensure the directory for logs exists — create if needed.
    """
    print(f"logfile: {log_full_path}")
    logs_directory = pathlib.Path(log_full_path).parent.absolute()
    print(f"Checking {logs_directory}")
    if not pathlib.Path(logs_directory).exists():
        print(
            "Directory does not exist at {0}, creating now...".format(
                str(logs_directory)
            )
        )
        logs_directory.mkdir(parents=True, exist_ok=True)


class TemplateSimulatorSession:
    def __init__(
        self,
        render: bool = False,
        env_name: str = "Ambo_Dispatcher",
        log_data: bool = False,
        log_file_name: str = None,
    ):
        """Simulator Interface with the Bonsai Platform

        Parameters
        ----------
        render : bool, optional
            Whether to visualize episodes during training, by default False
        env_name : str, optional
            Name of simulator interface, by default "Cartpole"
        log_data: bool, optional
            Whether to log data, by default False
        log_file_name : str, optional
            where to log data, by default None. If not specified, will generate a name.
        """

        self.simulator = environment.Env(**DEFAULT_CONFIG)
        self.env_name = env_name
        self.render = render
        self.log_data = log_data
        if not log_file_name:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file_name = current_time + "_" + env_name + "_log.csv"

        self.log_full_path = os.path.join(LOG_PATH, log_file_name)
        ensure_log_dir(self.log_full_path)
        self.sim_observation = self.simulator.reset()
        self._gym_terminal = False
        self._gym_reward = 0
        self.config = DEFAULT_CONFIG

    def get_state(self) -> Dict[str, float]:
        """Extract current states from the simulator

        Returns
        -------
        Dict[str, float]
            Returns float of current values from the simulator
        """
        print('sim_observations:', self.sim_observation)
        print('shape of dispatches:', len(self.sim_observation[0:-3]))
        return {
            # Add simulator state as dictionary
            "dispatches": np.array(self.sim_observation[0:-3]).tolist(),
            "location_x": float(self.sim_observation[-3]),
            "location_y": float(self.sim_observation[-2]),
            "time": float(self.sim_observation[-1]),
            "_gym_reward": float(self._gym_reward),
            "_gym_terminal": float(self._gym_terminal)
        }


    def halted(self) -> bool:
        """Halt current episode. Note, this should only return True if the simulator has reached an unexpected state.

        Returns
        -------
        bool
            Whether to terminate current episode
        """
        return False

    def episode_start(self, config: Dict = None) -> None:
        """Initialize simulator environment using scenario parameters from inkling.
        Note, `simulator.reset()` initializes the simulator parameters for initial positions and velocities of the cart and pole using a random sampler.
        See the source for details.

        Parameters
        ----------
        config : Dict, optional.
        """
        # Add simulator reset api here using config from desired lesson in inkling
        self.config.update(**config)
        # config to gym envs are not through reset
        # if needed config should part of gym properties and over-written before each reset
        self.sim_observation = self.simulator.episode_reset(**config)
        self.config_flattened = config.copy()
        
        if self.render:
            self.simulator.render()

    def episode_step(self, action: Dict):
        """Step through the environment for a single iteration.

        Parameters
        ----------
        action : Dict
            An action to take to modulate environment.
        """
        # Add simulator step api here using action from Bonsai platform
        self.sim_observation, self._gym_reward, self._gym_terminal, _ = self.simulator.step(
            action['dispatch_point'])

        if self.render:
            self.simulator.view()

    def log_iterations(
        self,
        state,
        action,
        episode: int = 0,
        iteration: int = 1,
    ):
        """Log iterations during training to a CSV.

        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        sim_speed_delay : float, optional
        """

        def add_prefixes(d, prefix: str):
            return {f"{prefix}_{k}": v for k, v in d.items()}

        # custom way to turn lists into strings for logging
        log_state = state.copy()
        log_action = action.copy()

        for key, value in log_state.items():
            if type(value) == list:
                log_state[key] = str(log_state[key])

        for key, value in log_action.items():
            if type(value) == list:
                log_action[key] = str(log_action[key])

        log_state = add_prefixes(log_state, "state")
        log_action = add_prefixes(log_action, "action")
        log_config = add_prefixes(self.config_flattened, "config")

        data = {**log_state, **log_action, **log_config}

        data["episode"] = episode
        data["iteration"] = iteration
        log_df = pd.DataFrame(data, index=[0])

        if os.path.exists(self.log_full_path):
            log_df.to_csv(
                path_or_buf=self.log_full_path, mode="a", header=False, index=False
            )
        else:
            log_df.to_csv(
                path_or_buf=self.log_full_path, mode="w", header=True, index=False
            )




def env_setup(env_file: str = ".env"):
    """Helper function to setup connection with Project Bonsai

    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    env_file_exists = os.path.exists(env_file)
    if not env_file_exists:
        open(env_file, "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(env_file, "SIM_WORKSPACE", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        set_key(env_file, "SIM_ACCESS_KEY", access_key)

    load_dotenv(verbose=True, override=True)
    workspace = os.getenv("SIM_WORKSPACE")
    access_key = os.getenv("SIM_ACCESS_KEY")

    return workspace, access_key

# Manual test policy loop


def test_policy(
    render=False,
    num_episodes: int = 30,
    num_iterations: int = 300,
    log_iterations: bool = False,
    policy=random_policy,
    policy_name: str = "random",
    scenario_file: str = "assess_config.json",
    exported_brain_url: str = "http://5200:5000"
):
    """Test a policy using random actions over a fixed number of episodes
    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """
    # Use custom assessment scenario configs
    with open(scenario_file) as fname:
        assess_info = json.load(fname)
    scenario_configs = assess_info['episodeConfigurations']
    num_episodes = len(scenario_configs)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name = current_time + "_" + policy_name + "_log.csv"
    sim = TemplateSimulatorSession(
        render=render,
        log_data=log_iterations,
        log_file_name=log_file_name
    )
    for episode in range(0, num_episodes):
        iteration = 1
        terminal = False
        # sim_state = sim.episode_start(config=default_config)
        sim_state = sim.episode_start(config=scenario_configs[episode-1])
        sim_state = sim.get_state()
        if log_iterations:
            action = policy(sim_state)
            for key, value in action.items():
                action[key] = None
            sim.log_iterations(sim_state, action, episode, iteration)
        print('------------------------------------------------------')
        print(f"Running iteration #{iteration} for episode #{episode}")
        iteration += 1
        while not terminal:
            action = policy(sim_state)
            sim.episode_step(action)
            sim_state = sim.get_state()
            if log_iterations:
                sim.log_iterations(sim_state, action, episode, iteration)
            print('------------------------------------------------------')
            print(f"Running iteration #{iteration} for episode #{episode}")
            iteration += 1
            terminal = iteration >= num_iterations+2 or sim.halted()
    return sim


def main(
    render: bool = False,
    log_iterations: bool = False,
    config_setup: bool = False,
    env_file: Union[str, bool] = ".env",
    workspace: str = None,
    accesskey: str = None,
):
    """Main entrypoint for running simulator connections

    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    config_setup: bool, optional
        if enabled then uses a local `.env` file to find sim workspace id and access_key
    env_file: str, optional
        if config_setup True, then where the environment variable for lookup exists
    workspace: str, optional
        optional flag from CLI for workspace to override
    accesskey: str, optional
        optional flag from CLI for accesskey to override
    """

    # check if workspace or access-key passed in CLI
    use_cli_args = all([workspace, accesskey])

    # check for accesskey and workspace id in system variables
    if all(
        [
            not use_cli_args,
            "SIM_WORKSPACE" in os.environ,
            "SIM_ACCESS_KEY" in os.environ,
        ]
    ):
        workspace = os.environ["SIM_WORKSPACE"]
        accesskey = os.environ["SIM_ACCESS_KEY"]
    elif use_cli_args:
        # use workspace and access key from CLI args passed into main
        pass
    elif config_setup or env_file:
        print(
            f"No system variables for workspace-id or access-key found, checking in env-file (.env by default)"
        )
        workspace, accesskey = env_setup(env_file)
        load_dotenv(verbose=True, override=True)
    else:
        pass

    # grab standardized way to interact with sim API
    sim = TemplateSimulatorSession(render=render, log_data=log_iterations)

    # configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # load json file as simulator integration config type file
    with open('interface.json') as file:
        interface = json.load(file)

    # create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=interface["timeout"],
        simulator_context=config_client.simulator_context,
        description=interface["description"],
    )

    def CreateSession(
        registration_info: SimulatorInterface, config_client: BonsaiClientConfig
    ):
        """Creates a new Simulator Session and returns new session, sequenceId
        """

        try:
            print(
                "config: {}, {}".format(
                    config_client.server, config_client.workspace)
            )
            registered_session: SimulatorSessionResponse = client.session.create(
                workspace_name=config_client.workspace, body=registration_info
            )
            print("Registered simulator. {}".format(
                registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print(
                "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
                    ex.status_code, ex.error.message, ex
                )
            )
            raise ex
        except Exception as ex:
            print(
                "UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(
                    ex
                )
            )
            raise ex

    registered_session, sequence_id = CreateSession(
        registration_info, config_client)
    episode = 0
    iteration = 0

    try:
        while True:
            # advance by the new state depending on the event type
            # TODO: it's risky not doing doing `get_state` without first initializing the sim
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
            )
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print(
                    "[{}] Last Event: {}".format(
                        time.strftime("%H:%M:%S"), event.type)
                )
            except HttpResponseError as ex:
                print(
                    "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                        ex.status_code, ex.error.message, ex
                    )
                )
                # this can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                # if your network has some issue, or sim session at platform is going away..
                # so let's re-register sim-session and get a new session and continue iterating. :-)
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                # ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                # if possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue

            # event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                episode += 1
            elif event.type == "EpisodeStep":
                sim.episode_step(event.episode_step.action)
                iteration += 1
                if sim.log_data:
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        action=event.episode_step.action,
                    )
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                iteration = 0
            elif event.type == "Unregister":
                print(
                    "Simulator Session unregistered by platform because '{}', Registering again!".format(
                        event.unregister.details
                    )
                )
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            else:
                pass
    except KeyboardInterrupt:
        # gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    # except Exception as err:
    #     # gracefully unregister for any other exceptions
    #     client.session.delete(
    #         workspace_name=config_client.workspace,
    #         session_id=registered_session.session_id,
    #     )
    #     print("Unregistered simulator because: {}".format(err))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Bonsai and Simulator Integration...")
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render training episodes",
    )
    parser.add_argument(
        "--log-iterations",
        action="store_true",
        default=False,
        help="Log iterations during training",
    )
    parser.add_argument(
        "--config-setup",
        action="store_true",
        default=False,
        help="Use a local environment file to setup access keys and workspace ids",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        metavar="ENVIRONMENT FILE",
        help="path to your environment file",
        default=".env",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        metavar="WORKSPACE ID",
        help="your workspace id",
        default=None,
    )
    parser.add_argument(
        "--accesskey",
        type=str,
        metavar="Your Bonsai workspace access-key",
        help="your bonsai workspace access key",
        default=None,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test-random", action="store_true",
    )

    group.add_argument(
        "--test-exported",
        type=int,
        const=5200,  # if arg is passed with no PORT, use this
        nargs="?",
        metavar="PORT",
        help="Run simulator with an exported brain running on localhost:PORT (default 5000)",
    )

    parser.add_argument(
        "--iteration-limit",
        type=int,
        metavar="EPISODE_ITERATIONS",
        help="Episode iteration limit when running local test.",
        default=200,
    )

    parser.add_argument(
        "--custom_assess",
        type=str,
        default=False,
        help="Custom assess config json filename",
    )

    args, _ = parser.parse_known_args()

    if args.test_random:
        test_policy(
            render=args.render, log_iterations=args.log_iterations, policy=random_policy
        )
    elif args.test_exported:
        port = args.test_exported
        url = f"http://localhost:{port}"
        print(f"Connecting to exported brain running at {url}...")
        scenario_file = 'machine_10_down.json'
        if args.custom_assess:
            scenario_file = args.custom_assess
        trained_brain_policy = partial(brain_policy, exported_brain_url=url)
        test_policy(
            render=args.render,
            log_iterations=args.log_iterations,
            policy=trained_brain_policy,
            policy_name="exported",
            num_iterations=args.iteration_limit,
            scenario_file=scenario_file
        )
    else:
        main(
            config_setup=args.config_setup,
            render=args.render,
            log_iterations=args.log_iterations,
            env_file=args.env_file,
            workspace=args.workspace,
            accesskey=args.accesskey,
        )
