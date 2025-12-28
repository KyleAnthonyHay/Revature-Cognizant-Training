# Model Format Comparison Report

| Format     | Size          | Contains              | Use Case           |
|------------|---------------|----------------------|-------------------|
| .keras     | 1.28 MB       | arch + weights + opt | Default choice    |
| .h5        | 1.28 MB       | arch + weights + opt | Legacy support    |
| SavedModel | 904.32 KB     | TF graph + vars      | TF Serving        |
| .weights   | 1.28 MB       | weights only         | Transfer learning |
