### spg_stocks is a project to build and deploy ML model using high-frequency finance data.

This project will predict returns in real time at 2-minute frequency.

Currently, I plan the following project dev cycle:
- v0.1: MVP, serves predictions, which are usually up-to-date.
- v0.2: More intuitive output text.
- v0.3: new webpage structure, predictions made the moment when end user clicks button
- v0.5: add new components: 
---- cloud storage for data and performance evalutaion logs
---- daily Cloud Function/ Cloud Schedulers to download one day of data and to run performance evale evaluation
- v1.0: add dashboard with performance evaluation
- v1.5: add model retraining every week (did not implement)
- v2.0: store data in BigQuery. add dashboard for feature distribution and train-serving skew
- v2.5: create performance-related alerts (did not implement)

Notes:
- when building v0.1 relying on the standard project dev/prod pipeline, need to do pip install --upgrade google-cloud-storage to make main.py work
- this app requires more RAM than previous ones, so I use F2, F4 or F4_1G GAE instance.cThe app is still running slow and often fails. At some point later, would be nice to fix that.

01/27: fixed a bug in performance evaluation CF. The bug led to failure of cf_perfeval due to large ram allocation.
        fixed it by optimizing a script for cf_perfeval to read only those daily files which are needed.
        this script was developed in a new repo (spg_stocks_test) on a new vm cpu. this was needed since old vm-cpu in unavailable most of time.
        
I am abandoning this project in favor of 'stocks_ete' repo. It implements this solution in production-grade Kubefloe pipeline.
I have stopped after implementing v1.0.



