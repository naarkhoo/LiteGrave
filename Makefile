run:
	streamlit run src/app.py

build-grobid:
	docker pull lfoppiano/grobid:0.8.0-arm

run-grobid:
	docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0-arm