target   := generated/int_mat_size.gen generated/e4m3_mat_size.gen generated/e4m3q_mat_size.gen

.PHONY: all

all: generated $(target)

generated:
	mkdir -p generated

generated/int_mat_size.gen: int_matgen.py
	python3 int_matgen.py

generated/e4m3_mat_size.gen: e4m3_matgen.py
	python3 e4m3_matgen.py

generated/e4m3q_mat_size.gen: e4m3q_matgen.py
	python3 e4m3q_matgen.py

clean:
	rm -rf generated