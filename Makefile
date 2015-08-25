all:
	clang++ -std=gnu++11 bilateral_filter.cpp -o bilateral_filter

debug:
	clang++ -g -std=gnu++11 bilateral_filter.cpp -o bilateral_filter

clean:
	rm bilateral_filter