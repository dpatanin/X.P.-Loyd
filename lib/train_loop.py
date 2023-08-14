import itertools
import multiprocessing
import queue
import signal
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Text, Tuple

import gymnasium

import lib.trackers as trackers_lib
import lib.types as types_lib


def run_env_loop(
    agent: types_lib.Agent, env: gymnasium.Env
) -> Iterable[
    Tuple[gymnasium.Env, types_lib.TimeStep, types_lib.Agent, types_lib.Action]
]:
    while True:  # For each episode.
        agent.reset()
        # Think of reset as a special 'action' the agent takes, thus given us a reward 'zero', and a new state 's_t'.
        observation = env.reset()
        reward = 0.0
        done = truncated = False
        first_step = True
        info = {}

        while True:  # For each step in the current episode.
            timestep_t = types_lib.TimeStep(
                observation=observation,
                reward=reward,
                done=done or truncated,
                first=first_step,
                info=info,
            )
            a_t = agent.step(timestep_t)
            yield env, timestep_t, agent, a_t

            a_tm1 = a_t
            observation, reward, done, truncated, info = env.step(a_tm1)

            first_step = False
            if done or truncated:  # Actual end of an episode
                # This final agent.step() will ensure the done state and final reward will be seen by the agent and the trackers
                timestep_t = types_lib.TimeStep(
                    observation=observation,
                    reward=reward,
                    done=True,
                    first=False,
                    info=info,
                )
                unused_a = agent.step(timestep_t)
                yield env, timestep_t, agent, None
                break


def run_env_steps(
    num_steps: int, agent: types_lib.Agent, env: gymnasium.Env, trackers: Iterable[Any]
) -> Mapping[Text, float]:
    seq = run_env_loop(agent, env)
    seq_truncated = itertools.islice(seq, num_steps)
    return trackers_lib.generate_statistics(trackers, seq_truncated)


def run_parallel_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    learner_agent: types_lib.Learner,
    eval_agent: types_lib.Agent,
    eval_env: gymnasium.Env,
    actor: types_lib.Agent,
    actor_env: gymnasium.Env,
    data_queue: multiprocessing.Queue,
    tb_log_dir: str,
) -> None:
    # Create shared iteration count and start, end training event.
    # start_iteration_event is used to signaling actors to run one training iteration,
    # stop_event is used to signaling actors the end of training session.
    # The start_iteration_event and stop_event are only set by the main process.
    start_iteration_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    # Run learner train loop on a new thread.
    learner = threading.Thread(
        target=run_learner,
        args=(
            num_iterations,
            num_eval_steps,
            learner_agent,
            eval_agent,
            eval_env,
            data_queue,
            start_iteration_event,
            stop_event,
            tb_log_dir,
        ),
    )
    learner.start()

    # Create and start actor processes once, this will preserve actor's internal state like steps etc.

    p = multiprocessing.Process(
        target=run_actor,
        args=(
            actor,
            actor_env,
            data_queue,
            num_train_steps,
            start_iteration_event,
            stop_event,
            tb_log_dir,
        ),
    )
    p.start()
    p.join()
    p.close()

    # Close queue.
    data_queue.close()


def run_actor(
    actor: types_lib.Agent,
    actor_env: gymnasium.Env,
    data_queue: multiprocessing.Queue,
    num_train_steps: int,
    start_iteration_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    tb_log_name: str = None,
) -> None:
    # Listen to signals to exit process.
    handle_exit_signal()

    actor_trackers = trackers_lib.make_default_trackers(f"{tb_log_name}/R2D2_actor")
    while not stop_event.is_set():
        # Wait for start training event signal, which is set by the main process.
        if not start_iteration_event.is_set():
            continue

        # Run training steps.
        run_env_steps(num_train_steps, actor, actor_env, actor_trackers)

        # Mark work done to avoid infinite loop in `run_learner_loop`,
        # also possible multiprocessing.Queue deadlock.
        data_queue.put("PROCESS_DONE")

        # Whoever finished one iteration first will clear the start training event.
        if start_iteration_event.is_set():
            start_iteration_event.clear()


def run_learner(
    num_iterations: int,
    num_eval_steps: int,
    learner: types_lib.Learner,
    eval_agent: types_lib.Agent,
    eval_env: gymnasium.Env,
    data_queue: multiprocessing.Queue,
    start_iteration_event: multiprocessing.Event,
    stop_event: multiprocessing.Event,
    tb_log_name: str = None,
) -> None:
    # Create trackers for learner and evaluator
    learner_trackers = trackers_lib.make_learner_trackers(f"{tb_log_name}/R2D2_learner")
    for tracker in learner_trackers:
        tracker.reset()

    eval_trackers = trackers_lib.make_default_trackers(f"{tb_log_name}/R2D2_eval")

    # Start training
    for _ in range(num_iterations):
        # Set start training event.
        start_iteration_event.set()
        learner.reset()

        run_learner_loop(learner, data_queue, learner_trackers)
        start_iteration_event.clear()

        # Run evaluation steps.
        run_env_steps(num_eval_steps, eval_agent, eval_env, eval_trackers)
        time.sleep(5)

    # Signal actors training session ended.
    stop_event.set()


def run_learner_loop(
    learner: types_lib.Learner,
    data_queue: multiprocessing.Queue,
    learner_trackers: Iterable[Any],
) -> None:
    """
    Run learner loop by constantly pull item off multiprocessing.queue and calls the learner.step() method.
    """

    is_actor_done = False

    # Run training steps.
    while True:
        # Try to pull one item off multiprocessing.queue.
        try:
            item = data_queue.get()
            is_actor_done = item == "PROCESS_DONE"
            if not is_actor_done:
                learner.received_item_from_queue(item)
        except queue.Empty:
            pass
        except EOFError:
            pass

        # Only break if actor process is done
        if is_actor_done:
            break

        # The returned stats_sequences could be None when call learner.step(), since it will perform internal checks.
        stats_sequences = learner.step()

        if stats_sequences is not None:
            for stats in stats_sequences:
                for tracker in learner_trackers:
                    tracker.step(stats)


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        sys.exit(128 + signal_code)

    # Listen to signals to exit process.
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
