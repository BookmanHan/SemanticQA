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
	virtual double probability(const tuple<int, int, int>& datum, const vector<int>& des_head, const vector<int>& des_tail) = 0;
	virtual int infer_entity(const vector<int>& des) = 0;
	virtual int infer_relation(const vector<int>& des) = 0;
	virtual void train_once(const tuple<int, int, int>& datum, const vector<int>& des_head, const vector<int>& des_tail) = 0;
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
	virtual void train_kernel()
	{
//#pragma omp parallel for
		for (auto i = dm.knowledge.begin();   i != dm.knowledge.end(); ++i)
		{
			train_once(*i, dm.description[get<0>(*i)], dm.description[get<2>(*i)]);
		}
	}

	void train(const int epos, function<void(int)> fn_round_check = [&](int epos){})
	{
		logout.record() << "Start to Train";

		for (auto ind_turn = 0; ind_turn < epos; ++ind_turn)
		{
			logout.record() << "Round @ " << ind_turn;

			train_kernel();

			logout << " Finished";

			fn_round_check(ind_turn);
			logout.flush();
		}

		logout.record() << "Train Over";
	}
};