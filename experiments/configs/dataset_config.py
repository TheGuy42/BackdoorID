from src.data.dataset_base import ModelDatasetConfig
from src.utils.base_config import BaseConfig
from src.data.BAIT_dataset import BAITDatasetConfig
from src.data.PADBench_dataset import PADBenchDatasetConfig


class ExpDatasetConfig(BaseConfig):
    train_dataset_config: list[ModelDatasetConfig]
    val_dataset_config: list[ModelDatasetConfig] | None = None
    test_dataset_configs: dict[str, ModelDatasetConfig]

EXP_DATASET_CONFIGS:dict[str, ExpDatasetConfig] = {} 

BAIT_ROOT = "datasets/BAIT-ModelZoo"
PADBENCH_ROOT = "datasets/PADBench"

################################################################
############# Experiment Dataset Configurations ################
################################################################

EXP_DATASET_CONFIGS['bait_llama2_7b'] = ExpDatasetConfig(
    train_dataset_config = [BAITDatasetConfig(
        dataset_root=BAIT_ROOT,
        base_models=["meta-llama/Llama-2-7b-hf"],
        model_ids=['id-0005', 'id-0072', 'id-0074', 'id-0024', 'id-0027', 'id-0084', 'id-0014', 'id-0076'],
    )],
    val_dataset_config=[BAITDatasetConfig(
        dataset_root=BAIT_ROOT,
        base_models=["meta-llama/Llama-2-7b-hf"],
        model_ids=['id-0089', 'id-0047', 'id-0053', 'id-0048', 'id-0006', 'id-0060', 'id-0063', 'id-0039', 'id-0044', 'id-0011', 'id-0008'],
    )],
    test_dataset_configs = {
        "BAIT_llama2_Test": BAITDatasetConfig(
            dataset_root=BAIT_ROOT,
            base_models=["meta-llama/Llama-2-7b-hf"],
            model_ids=['id-0034', 'id-0068', 'id-0065', 'id-0028', 'id-0052', 'id-0031', 'id-0010', 'id-0054', 'id-0009', 'id-0043', 'id-0040', 'id-0067'],
        ),
    }
)

EXP_DATASET_CONFIGS['bait_llama3_8b'] = ExpDatasetConfig(
    train_dataset_config = [BAITDatasetConfig(
        dataset_root=BAIT_ROOT,
        base_models=["meta-llama/Meta-Llama-3-8B-Instruct"],
        model_ids=['id-0020', 'id-0036', 'id-0086', 'id-0004', 'id-0090', 'id-0058', 'id-0087', 'id-0019'],
    )],
    val_dataset_config=[BAITDatasetConfig(
        dataset_root=BAIT_ROOT,
        base_models=["meta-llama/Meta-Llama-3-8B-Instruct"],
        model_ids=['id-0035', 'id-0059', 'id-0023', 'id-0000', 'id-0022', 'id-0042', 'id-0070', 'id-0018', 'id-0061', 'id-0049'],
    )],
    test_dataset_configs = {
        "BAIT_llama3_Test": BAITDatasetConfig(
            dataset_root=BAIT_ROOT,
            base_models=["meta-llama/Meta-Llama-3-8B-Instruct"],
            model_ids=['id-0015', 'id-0055', 'id-0045', 'id-0032', 'id-0029', 'id-0081', 'id-0073', 'id-0025', 'id-0051', 'id-0037', 'id-0030', 'id-0056'],
        ),
    }
)

EXP_DATASET_CONFIGS['bait_mistral_7b'] = ExpDatasetConfig(
    train_dataset_config = [BAITDatasetConfig(
        dataset_root=BAIT_ROOT,
        base_models=["mistralai/Mistral-7B-Instruct-v0.2"],
        model_ids=['id-0003', 'id-0064', 'id-0080', 'id-0002', 'id-0088', 'id-0078', 'id-0082', 'id-0012'],
    )],
    val_dataset_config=[BAITDatasetConfig(
        dataset_root=BAIT_ROOT,
        base_models=["mistralai/Mistral-7B-Instruct-v0.2"],
        model_ids=['id-0033', 'id-0085', 'id-0057', 'id-0001', 'id-0016', 'id-0069', 'id-0077', 'id-0007', 'id-0075', 'id-0071'],
    )],
    test_dataset_configs = {
        "BAIT_mistral_Test": BAITDatasetConfig(
            dataset_root=BAIT_ROOT,
            base_models=["mistralai/Mistral-7B-Instruct-v0.2"],
            model_ids=['id-0013', 'id-0066', 'id-0046', 'id-0062', 'id-0026', 'id-0083', 'id-0079', 'id-0021', 'id-0038', 'id-0050', 'id-0017', 'id-0041'],
        ),
    }
)

EXP_DATASET_CONFIGS['RQ3'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        ),
        PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_13b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        ),
        PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["qwen1.5_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        ),
        PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_13b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        ),
        PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["qwen1.5_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "toxic_backdoors_hard_llama2_7b[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
        "toxic_backdoors_hard_llama2_13b[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_13b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
        "toxic_backdoors_hard_qwen1.5_7b[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["qwen1.5_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['toxic_backdoors_alpaca'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_alpaca"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_alpaca"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "toxic_backdoors_alpaca[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_alpaca"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['toxic_backdoors_hard_qwen'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["qwen1.5_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["qwen1.5_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "toxic_backdoors_hard_qwen1.5_7b[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["qwen1.5_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['toxic_backdoors_hard_llama2_7b'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "toxic_backdoors_hard_llama2_7b[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['toxic_backdoors_hard_llama2_13b'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_13b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_13b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "toxic_backdoors_hard_llama2_13b[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_13b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['toxic_backdoors_hard_llama3_8b'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama3_8b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama3_8b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "toxic_backdoors_hard_llama3_8b[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama3_8b"],
            datasets=["toxic_backdoors_hard"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['squad'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["squad_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["squad_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "squad[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["squad_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['imdb_insertsent'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "imdb_insertsent[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['imdb_stybkd'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "imdb_stybkd[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['imdb_RIPPLE'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "imdb_RIPPLE[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['imdb_syntactic'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_syntactic"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_syntactic"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "imdb_syntactic[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["imdb_syntactic"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['ag_news_insertsent'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "ag_news_insertsent[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['ag_news_insertsent_ext'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)]+[str(i) for i in range(75,100)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "ag_news_insertsent[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_insertsent"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['ag_news_stybkd'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "ag_news_stybkd[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['ag_news_stybkd_ext'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)]+[str(i) for i in range(75,100)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "ag_news_stybkd[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_stybkd"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['ag_news_RIPPLE'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "ag_news_RIPPLE[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['ag_news_RIPPLE_ext'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)]+[str(i) for i in range(75,100)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "ag_news_RIPPLE[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_RIPPLE"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)

EXP_DATASET_CONFIGS['ag_news_syntactic'] = ExpDatasetConfig(
    train_dataset_config = [PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_syntactic"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25)],
        )],
    val_dataset_config=[PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_syntactic"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(25, 50)],
        )],
    test_dataset_configs = {
        "ag_news_syntactic[50-75]": PADBenchDatasetConfig(
            dataset_root=PADBENCH_ROOT,
            base_models=["llama2_7b"],
            datasets=["ag_news_syntactic"],
            ranks=["rank256"],
            adapter_ids=[str(i) for i in range(50, 75)],
        ),
    },
)
