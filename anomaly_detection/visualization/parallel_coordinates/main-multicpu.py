from visualization.parallel_coordinates.pc import PC
from multiprocessing import Pool

filenames = ["scenario_1.binetflow"]#, "scenario_2.binetflow", "scenario_6.binetflow", "scenario_8.binetflow", "scenario_9.binetflow"]
# filenames = ["small_sample1.csv", "small_sample2.csv"]

datasets_path = "../../../datasets/"
output_path = "../output/parallel-coordinates"

analyzed_features = [
    # "StartTime", # timestamp, all will be unique, no point in analyzing that
    "Dur",
    "Proto", # transform to numerical, not all describe feature are useful
    "SrcAddr", # probably not useful
    "Sport", # transform to numerical, not all describe feature are useful
    "Dir", # transform to numerical, not all describe feature are useful
    "DstAddr", # probably not useful
    "Dport", # transform to numerical, not all describe feature are useful
    "State", # transform to numerical, not all describe feature are useful
    # "sTos",
    # "dTos",
    "TotPkts",
    "TotBytes",
    "SrcBytes"
    # "inlier"
]


pc = PC(datasets_path=datasets_path, output_path=output_path, analyzed_features=analyzed_features)

if __name__ == '__main__':
    pool = Pool()
    pool.map(pc.draw, filenames)
    pool.close()
    pool.join()
