# Define your parameter lists
set min_cluster_sizes 80 60 100
set min_samples_list 2 1 3
set cl_methods eom leaf
set cl_eps_list 0.04 0.05 0.06
set n_neighbors 50 60 40
# Loop over all combinations
for min_cluster_size in $min_cluster_sizes
    for min_samples in $min_samples_list
        for cl_method in $cl_methods
            for cl_eps in $cl_eps_list
                for n_n in $n_neighbors
                    echo "Running with: min_cluster_size=$min_cluster_size min_samples=$min_samples cl_method=$cl_method cl_eps=$cl_eps "

                    docker run --rm \
                        -v (pwd)/src:/app/src \
                        -v (pwd)/data:/app/data \
                        -v (pwd)/outputs:/app/outputs \
                        binclust:latest --reducer umap \
                        --clusterer hdbscan \
                        --hdb-min-cluster-size $min_cluster_size \
                        --hdb-min-samples $min_samples \
                        --hdb-cluster-selection-method $cl_method \
                        --hdb-cluster-selection-epsilon $cl_eps \
                        --umap-n-neighbors $n_n \
                        --mlflow \
                        --mlflow-experiment umap-hdbscan4
                end
            end
        end
    end
end

docker run --rm \
    -v (pwd)/src:/app/src \
    -v (pwd)/data:/app/data \
    -v (pwd)/outputs:/app/outputs \
    binclust:latest --reducer umap \
    --clusterer hdbscan \
    --hdb-min-cluster-size 60 \
    --hdb-min-samples 1 \
    --hdb-cluster-selection-method eom \
    --hdb-cluster-selection-epsilon 0.06 \
    --umap-n-neighbors 60 \
    --epsilon-cover-enabled \
    --epsilon-use-cluster-scope \
    --epsilon-cover-plot