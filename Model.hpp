#pragma once
#include "Import.hpp"
#include "DataSet.hpp"
#include "DataModel.hpp"
#include "File.hpp" 
#include "Logging.hpp"

class Model
{
public:
	DataModel dm;

public:
	Model(const DataSet& ds)
		:dm(ds)
	{
		;
	}

public:
	virtual double probability(const tuple<int, int, int>& datum) = 0;
	virtual void train_knowledge(const tuple<int, int, int>& datum) = 0;
	virtual void save(FormatFile & file) = 0;
	virtual void load(FormatFile & file) = 0;

public:
	void save(const string& path)
	{
		FormatFile& file = make_fout(path);
		save(file);
		file.close();
	}

	void load(const string& path)
	{
		FormatFile& file = make_fin(path);
		load(file);
		file.close();
	}

public:
	virtual void train_kernel(function<void()> fn_middle_process = [&](){})
	{
#pragma omp parallel for
		for (auto i = dm.knowledge.begin(); i != dm.knowledge.end(); ++i)
		{
			train_knowledge(*i);
		}
	}

	void train(const int epos, function<void(int)> fn_round_check = [&](int epos){})
	{
		logout.record() << "Start to Train";

		for (auto ind_turn = 0; ind_turn < epos; ++ind_turn)
		{
			logout.record() << "Round @ " << ind_turn;
			logout.flush();

			train_kernel();

			logout << " Finished";

			fn_round_check(ind_turn);
			logout.flush();
		}

		logout.record() << "Train Over";
	}

public:
	void test()
	{
		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double rmrr = 0;
		double fmrr = 0;
		int total = 0;

#pragma omp parallel for
		for (auto i = dm.knowledge.end() - 16*6; i!= dm.knowledge.end(); ++i)
		{
			tuple<int, int, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = probability(*i);

			for (auto j = 0; j != dm.n_entity; ++j)
			{
				get<2>(t) = j;

				if (score_i >= probability(t))
					continue;

				++rmean;

				if (dm.sampling_checker.find(t) == dm.sampling_checker.end())
				{
					++frmean;
				}
			}

#pragma omp critical
			{
				mean += rmean;
				fmean += frmean;
				rmrr += 1.0 / (rmean + 1);
				fmrr += 1.0 / (frmean + 1);

				if (rmean < 10)
					++hits;
				if (frmean < 10)
					++fhits;
			}
		}

		logout.record() << "[Test] fHITS@10 = " << fhits / (96 * 6);

		std::cout.flush();
	}
};