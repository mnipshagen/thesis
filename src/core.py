from thesis.model import models
from thesis.preprocessing import preprocess as pp
from thesis.train import trainer

if __name__ == "__main__":
    """
    Run the pipeline: Preprocess & Aggregate -> Train
    """
    # TODO: create commandline input
    # TODO: create plot and eval pipeline
    result = pp.main(
        redo=False,
        include_voctrain=True,
        verbose=True,
        original_data=True
        # valid_ids_path="./data/interim/valid_ids",
        # data_dict_path="./data/interim/data_dict",
        # agg_data_dict_path="./data/interim/data_dict_agg",
        # output_df_path="./data/out/data"
    )
    Y = result["ts15218"]
    result.drop(columns=["ts15218"], inplace=True)

    trainer.train(result, Y, subset=["rfc", "dtc", "svm"])