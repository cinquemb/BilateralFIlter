#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>

/*
re: https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/_denoise_cy.pyx
and https://en.wikipedia.org/wiki/Bilateral_filter
						pad		original 	pad
constant (with c=0):  0 0 0 0 | 1 2 3 4 | 0 0 0 0
wrap: 				  1 2 3 4 | 1 2 3 4 | 1 2 3 4
symmetric: 		  	  4 3 2 1 | 1 2 3 4 | 4 3 2 1
edge: 				  1 1 1 1 | 1 2 3 4 | 4 4 4 4
reflect:			  3 4 3 2 | 1 2 3 4 | 3 2 1 2
*/

long double gaussian_weight(long double& sigma, long double& value){
	return std::exp(-0.5 * std::pow((value / sigma),2));
}

int mod (int a, int b)
{
   int ret = a % b;
   if(ret < 0)
     ret+=b;
   return ret;
}

long double calculate_stdv(std::vector<long double>& v, long double& _mean){
    long double stdv_exp = 2;
    long double _accum = 0.0;
    std::for_each (v.begin(), v.end(), [&](const long double d) {
        _accum += std::pow((d - _mean), stdv_exp);
    });
    long double _stdev = std::sqrt(_accum/ ((long double)v.size()-1));
    return _stdev;
}


std::vector<long double> bilateral_filter1d(std::vector<long double>&ts, long double& color_range_sigma, int& spacial_range_sigma_factor, std::string& wrap){
	long double ts_min = (long double)std::numeric_limits<int>::max();
	long double ts_max = (long double)std::numeric_limits<int>::min();
	for(int i =0;i<ts.size();++i){
		if(ts_min > ts[i])
			ts_min = ts[i];
		if(ts_max < ts[i])
			ts_max = ts[i];
	}
	std::transform(ts.begin(), ts.end(), ts.begin(), std::bind1st(std::multiplies<double>(),std::abs(1/ts_max)));
	
	long double normalize_color_denom = ts_max-ts_min;
	long double ts_sum = std::accumulate(ts.begin(), ts.end(), 0.0);
	int ts_len = ts.size();
    long double ts_mean =  ts_sum / (long double)ts_len;
    long double ts_stdv = calculate_stdv(ts, ts_mean);
	long double spacial_range_sigma = spacial_range_sigma_factor+1;
	int window_size = (int)std::ceil(2.5*spacial_range_sigma);
	std::vector<long double> bilater_filter1d_data(ts.size(),0.0);

	for(int i=0;i<ts_len;++i){
		long double numerator_sum = 0.0;
		long double denominator_sum = 0.0;
		int window_range = (window_size*2)+1;
		std::vector<long double> w_ts(window_range,0);

		for(int j=0;j<w_ts.size();++j){
			if(wrap == "reflect"){
				int w_index = mod(i+(j-window_size), ts_len);
				w_ts[j] = ts[w_index];
			}else if(wrap == "constant"){
				if((i+(j-window_size) < ts_len) && (i+(j-window_size) > -1))
					w_ts[j] = ts[(i+(j-window_size))];
				else
					w_ts[j] = 0;
			}else if(wrap == "edge"){
				if((i+(j-window_size) < ts_len) && (i+(j-window_size) > -1))
					w_ts[j] = ts[(i+(j-window_size))];
				else if((i+(j-window_size) < 0))
					w_ts[j] = ts[0];
				else if((i+(j-window_size) >= ts_len))
					w_ts[j] = ts[ts_len-1];
			}

			long double ts_c = (ts[i]-ts_mean)/(ts_stdv);
			long double w_ts_c = (w_ts[j]-ts_mean)/(ts_stdv);

			long double fs_d = (std::sqrt(std::pow((w_ts_c- ts_c), 2)+ std::pow((((j-window_size)/(long double)window_size)), 2)));
			long double fs = gaussian_weight(color_range_sigma, fs_d);

			long double gs_d = std::sqrt(std::pow((w_ts[j]-ts[i]), 2) + std::pow((j-window_size), 2));
			long double gs = gaussian_weight(spacial_range_sigma, gs_d);

			denominator_sum += fs*gs;
			numerator_sum += w_ts[j] * fs*gs;
		}

		long double filtered_val = (numerator_sum/denominator_sum);
		bilater_filter1d_data[i] += (filtered_val * std::abs(ts_max));
	}
	return bilater_filter1d_data;
}

int main(int argc, char *argv[]){
	std::vector<long double> time_series = {0.,-0.27267267057145633,-0.6672983789995302,-0.5338541947930846,-0.6117404489279314,-0.6755527076595494,-0.02125421294486496,-0.10792797291843935,-0.6138271235477938,-0.3248568606554575,-0.08843449054916136,-0.27959358731026884,-0.0888007291071714,0.1415063448524414,0.3945869158380465,0.42460445574418765,0.19538043144765438,0.18794825506078514,0.4466366371456647,0.7412067594673032,0.702020217881393,0.929384768531743,0.7160918501006694,1.1367473444629899,0.9455116410606706,0.6816110508549922,0.9284369990611887,0.7368522856916324,0.7016688423746514,0.7716960777000934,0.6696174869530034,0.3413366356437346,0.6222068746258664,1.115103296314755,1.2670384896729314,0.9096106386086915,1.0324426751224536,0.5005217351302846,0.40480654024763996,0.4385674149307453,0.5884868769419367,0.5156528142940698,0.42148871556843104,-0.030282655987769358,-0.4294272076827578,-1.1499816593578638,-1.7917212167753855,-2.0278755948940943,-2.03218716031428,-2.4472502106563234,-2.25699859700199,-1.8585162619358127,-2.2768006264234932,-2.338886781890066,-2.2125605389328933,-2.2349224573435538,-1.90525983378493,-2.2391807388758718,-2.2288539070468096,-2.2378979046878213,-1.5329197713720863,-1.333021520594421,-0.8426121066244934,-0.2958742741381325,0.1820685074538087,0.3351358494895932,-0.27811480695856106,-0.37881258236164755,-0.3136908450507813,-0.2523429462249961,-0.22434188921077255,-0.016247896441508147,-0.055802975682963024,0.11964002782556127,-0.04911322903351359,-0.03468952471055063,-0.15768669544360087,-0.3339502518201483,-1.0829433618348543,-0.8955426026980875,-0.7744459810295016,-0.6934135084794563,-0.6049372025236891,-0.3518428861135055,0.09381114243163546,-0.12306059246134199,0.15196786433793408,0.21058839335616136,-0.029107789922687283,-0.23266852965087914,-0.227190840221013,-0.3489496060473642,-0.6597750655163467,-0.9340786420151763,-1.0468886239472694,-0.6967351187348036,-0.4506028218848533,-0.5653904542053424,-0.49980440691873723,-0.5388679788215971,-0.4101602764645551};
	std::vector<long double> sdsd = {0.1,0.5};
	for (auto is = sdsd.begin(); is != sdsd.end(); ++is){
		for(int i=1;i<11;++i){
			long double color_range_sigma = *is;
			int spacial_range_sigma_factor = i;
			std::string wrap = "reflect";

			std::vector<long double> bf1d = bilateral_filter1d(time_series, color_range_sigma, spacial_range_sigma_factor, wrap);
			std::cout.precision(12);
			for (auto it = bf1d.begin(); it != bf1d.end(); ++it)
				std::cout << *it << ",";
			std::cout << '\n';
		}
	}
}