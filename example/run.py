import os
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="verifai.error_table")
from verifai.samplers import ScenicSampler
from verifai.falsifier import generic_falsifier
from verifai.scenic_server import ScenicServer
from verifai.monitor import specification_monitor
from dotmap import DotMap


class SpeedSpec(specification_monitor):
    def __init__(self, csv):
        self.trace_id = 0
        self.csv = csv
        def spec(sim_result):
            ego_vx = sim_result.records.get("ego_vx", [])
            ego_vy = sim_result.records.get("ego_vy", [])
            ego_position = sim_result.records.get("ego_position", [])
            
            # Extract values
            ego_x = list(map(lambda x: x[-1][0], ego_position))
            ego_y = list(map(lambda x: x[-1][1], ego_position))
            ego_vx = list(map(lambda x: x[-1], ego_vx))
            ego_vy = list(map(lambda x: x[-1], ego_vy))
            
            # Log final values
            trace = {
                "id" : self.trace_id,
                "ego_x": ego_x,
                "ego_y": ego_y,
                "ego_vx": ego_vx,
                "ego_vy": ego_vy,
            }

            self.trace_id += 1

            df = pd.DataFrame(trace)
            df.to_csv(self.csv, mode='a', header=not os.path.exists(self.csv), index=False)
            
            return True
        
        super().__init__(spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run falsification for a single Scenic scenario.")
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenic file"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of falsification iterations"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Csv file for saving traces"
    )
    args = parser.parse_args()

    if os.path.exists(args.csv):
        os.remove(args.csv)
        print(f"{args.csv} deleted.")

    sampler = ScenicSampler.fromScenario(args.scenario)

    params = DotMap(
        n_iters=args.iters,
        save_error_table=True,
    )

    falsifier = generic_falsifier(
        sampler=sampler,
        monitor=SpeedSpec(csv=args.csv),
        falsifier_params=params,
        server_class=ScenicServer
    )

    falsifier.run_falsifier()
    print(f"Model checking completed for scenario: {args.scenario}")

