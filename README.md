<h1>Sequential Recommendation (KNN vs. BERT4Rec)</h1>

This is the code and the data used to conduct the experiments in the paper
titled <i>"Sequential Recommendation: A Study on Transformers, Nearest Neighbors and Sampled Metrics"</i>. It implements sequential KNN methods (VSKNN_SEQ and VSTAN_SEQ) and
includes the code for the BERT4Rec model (with minor changes) that is developed and shared by:
<br>
Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang,
BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer,
CIKM 2019. (<a href="https://github.com/FeiSun/BERT4Rec">Original Code</a>) 
<br><br>
Note: this version will be shared publicly later (on GitHub).
<h2>Related Datasets</h2>
Datasets can be downloaded from: <a href="https://drive.google.com/drive/folders/13r5G6Tb5u0Yb3fQjkriPGeR9W0A16M6P?usp=sharing">https://drive.google.com/drive/folders/13r5G6Tb5u0Yb3fQjkriPGeR9W0A16M6P?usp=sharing</a>
<h2>Requirements</h2>
To run the KNN method, the following libraries are required:

<ul>
    <li>Anaconda 4.X (Python 3.5)</li>
    <li>NumPy</li>
    <li>Pandas</li>
    <li>Six</li>
    <li>Pympler</li>
    <li>Python-dateutil</li>
    <li>Pytz</li>
    <li>Pyyaml</li>
    <li>Python-telegram-bot</li>
</ul>

Requirements for running BERT4Rec:

<ul>
    <li>Python 3.6+</li>
    <li>Tensorflow 1.14 (GPU version)</li>
    <li>CUDA compatible with TF 1.14</li>
</ul>

<h2>Installation</h2>
<h3>Using Docker </h3>
<ol>
    <li>Download and install Docker (https://www.docker.com/)</li>
    <li>Run the following commands: <code>docker pull recommendersystems/sequence-aware-image:version1</code></li>
</ol>

<h3>Using Anaconda</h3> 
<ol>
    <li>Download and install Anaconda (https://www.anaconda.com/distribution/)</li>
    <li>In the main folder, run the following commands:
        <ol>
            <li><code>conda install --yes --file requirements_conda.txt</code></li>
            <li><code>pip install -r requirements_pip.txt</code></li>      
        </ol>
    </li>
</ol>
<h2>How to Run It</h2>
<ol>
    <li><h4>Dataset preprocessing for KNNs (considering only orders)</h4>
    <ol>
        <li>
            Unzip and add any preprocessed dataset file for BERT4Rec (provided in its original source code)
            to the data folder, e.g., add <i>beauty.txt</i> to the folder <i>data</i>
        </li>
        <li>
            Run the one of the <code>gen_data_baselines_extended.py</code> or <code>gen_data_baselines_seq.py</code> files, for example:
            <ol>
                <li>
                For the Beauty dataset, where we keep each user history as one sequence (without sessionization),
                run the following command:
                <br>
                <code>
                python gen_data_baselines_extended.py --dataset_name=beauty --signature=baselines
                </code>
                </li>
                <li>
                For the other datasets, where we adopt sessionization to split user histories into sessions
                of size <i>L</i> with <i>P%</i> overlap, run the following command (this example is for the ML-1m dataset):
                <br>
                <code>
                python gen_data_baselines_seq.py --dataset_name=ml-1m --train_length=25 --test_length=25 --overlap=50 --signature=25_25_50
                </code>
                </li>
            </ol>
        </li>
    </ol>
    </li>
    <li><h4>Dataset preprocessing for KNNs (considering timestamps)</h4>
    <ol>
        <li>
            Unzip and add any original (raw) dataset file to the data folder,
            e.g., <i>ratings_Beauty.csv</i>  to the folder <i>data/raw</i>
        </li>
        <li>
            Run the related preprocessing file to create
            the same dataset as the prepred one for BERT4Rec
            but including timestamps, for example run the following command:
            <br>
            <code>
                python preprocess_beauty.py
            </code>
        </li>
        <li>
            The resulting file of the previous step will be stored in the folder <i>data</i>,
            for example <i>beauty_timestamps.txt</i>.
            <ol>
                <li>
                For the Beauty dataset, where we keep each user history as one sequence (without sessionization),
                run the following command:
                <code>
                python gen_data_baselines_extended_timestamps.py --dataset_name=beauty_timestamps --signature=baselines
                </code>  
                </li>
                <li>
                For the other datasets, where we adopt sessionization to split user histories into sessions
                of size <i>L</i> with <i>P%</i> overlap, run the following command (this example is for the ML-1m dataset):
                <br>
                <code>
                python gen_data_baselines_seq_timestamps.py --dataset_name=ml-1m_timestamps --train_length=25 --test_length=25 --overlap=50 --signature=25_25_50
                </code>
                </li>
            </ol>            
        </li>
    </ol>
    <li><h4>Preparing sampled version of the ML-20m dataset</h4>
    <ol>
        <li>
            Unzip and add any dataset file to the data folder,
            e.g., add <i>ratings_Beauty.csv</i> to the folder <i>data</i>
        </li>
        <li><h5>For BERT4Rec</h5>
            Define a percentage of the users that should be included in the sampled version of the dataset by
            <i>signature</i> parameter. Run the following command to create a sampled version of the dataset for
            BERT4Rec (this example is for the Ml-20m dataset):
            <br>
            <code>
                python data_sampled_bert4rec.py --dataset_name=ml-20m --signature=5
            </code>
        </li>
        <li><h5>For KNNs (considering only orders)</h5>
            Define a percentage of the users that should be included in the sampled version of the dataset by
            <i>sample</i> parameter.
            For datasets with sessionization, run the following command to create a sampled version of the
            dataset (this example is for the ML-20m dataset):
            <br>
            <code>
                python gen_data_baselines_seq_sample.py --dataset_name=ml-20m --sample=5 --train_length=20 --test_length=20 --overlap=50 --signature=5_20_20_50
            </code>
        </li>
        <li><h5>For KNNs (considering timestamps)</h5>
            Define a percentage of the users that should be included in the sampled version of the dataset by
            <i>sample</i> parameter.
            <br>
            <code>
                python gen_data_baselines_seq_timestamps_sample.py --dataset_name=ml-20m_timestamps --sample=5 --train_length=20 --test_length=20 --overlap=50 --signature=5_20_20_50
            </code>
        </li>
    </ol>
    </li>
    </li>
    <li><h4>Run experiments for KNNs (VSKNN_SEQ & VSTAN_SEQ)</h4>
    <ol>
        <li>
            Take the related configuration file <b>*.yml</b> from folder <i>KNN/conf/saved</i>, and
            put it into the folder named <i>KNN/conf/in</i>. It is possible to put multiple configure files
            in the <i>KNN/conf/in</i> folder. When a configuration file in <i>KNN/conf/in</i> has been executed,
            it will be moved to the folder <i>KNN/conf/out</i>.
        </li>
        <li>
            Run the following command from the main folder: </br>
            <br>
            <code>
                python run_config.py KNN/conf/in KNN/conf/out
            </code></br>
            If you want to run a specific configuration file, run the following command for example:</br>
            <br>
            <code>
                python run_config.py KNN/conf/saved/timestamps/exact_evaluation/exp/beauty/exp_beauty_vstan_seq.yml
            </code>
        </li>    
        <li>
            Results will be displayed and saved to the results folder as config.
        </li>
    </ol>
    </li>
    <li><h4>Run experiments for BERT4Rec</h4>
    Run the related bash shell script file under the main folder.
            The name of the file includes <i>(i)</i> the name of the dataset,
            <i>(ii)</i> the size of the network (hidden_size),
            <i>(iii)</i> the way of collecting the metrics 
            (<b>exact:</b> exact metrics,
            <b>pop</b>: sampled metrics using popularity-based sampling, and
            <b>uniform</b>: sampled metrics using random sampling based on uniform distribution), and
            <i>(iv)</i> the sample size for sampled metrics (100 or 1000). 
            For example, to run the experiment on the ML-1m dataset using exact evaluation 
            run the following script:
            <br>
            <code>
                run_ml-1m_256_exact.sh
            </code>
    </li>
</ol>


<h2>How to Configure It</h2>
Start from one of the configuration files under the <i>KNN/conf/saved</i> folder.
<h3>Essential Options</h3>
<div>
<div>
    <table class="table table-hover table-bordered">
        <tr>
            <th width="12%" scope="col"> Entry</th>
            <th width="16%" class="conf" scope="col">Example</th>
            <th width="72%" class="conf" scope="col">Description</th>
        </tr>
        <tr>
            <td>type</td>
            <td>single</td>
            <td>Values: <b>single</b> (for running experiments), <b>opt</b> (for tuning hyperparameters).
            </td>
        </tr>
        <tr>
            <td>evaluation</td>
            <td>evaluation_last</td>
            <td>Value: <b>evaluation_last</b> (for leave-last-out evaluation)
            </td>
        </tr>
        <tr>
            <td scope="row">negative_sampling_mode</td>
            <td>True</td>
            <td>The way of collecting metrics.<br>
            Values: <b>True</b> (for sampled metrics), <b>False</b> (for exact metrics).
            </td>
        </tr>
        <tr>
            <td scope="row">sampling_conf</td>
            <td>pop_random: True<br> sample_size: 1000</td>
            <td>Sampling strategy and sample size in sampled metrics.<br>
            Values for pop_random: <b>True</b> (for popularity_based sampling), <b>False</b> (for uniform sampling).
            </td>
        </tr>
        <tr>
            <td scope="row">optimize</td>
            <td> iterations: 100  # optional</td>
            <td>The number of iterations for random hyperparameter optimization.<br>
            </td>
        </tr>
        <tr>
            <td scope="row">algorithms</td>
            <td>-</td>
            <td>See the configuration files in the <i>KNN/conf/saved</i> folder for the KNNs
                algorithms and their parameters.<br>
            </td>
        </tr>
    </table>
</div>
</div>