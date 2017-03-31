#pragma once
#include "Import.hpp"
#include "DataSet.hpp"
#include "DataModel.hpp"
#include "File.hpp"
#include "Logging.hpp"
#include "Model.hpp"
#include <boost/progress.hpp>

class Task
{
protected:
	const DataModel& dm;

protected:
	double mean_bias;
	double bound_max;
	double bound_min;

public:
	Model* model;

public:
	Task(Model* model)
		:dm(model->dm), model(model)
	{
		;
	}

	Task(DataModel& dm)
		:dm(dm)
	{
		;
	}

	void set_model(Model* modelt)
	{
		this->model = modelt;
	}

public:
	virtual void test() = 0;
	virtual void result() = 0;
};

class EntityAnswering
	:public Task
{
public:
	struct Performance
	{
		double bm_accu;
	} best, current;

public:
	EntityAnswering(Model* dm)
		:Task(dm)
	{
		best.bm_accu = 0;
	}

public:
	virtual void test() override
	{
		logout.record() << "[Task]\tEntity Answering";

		current.bm_accu = 0;
		int n_TOTAL = 1000;

		current.bm_accu /= n_TOTAL;

		string inc_mark_hit = "";
		if (best.bm_accu <= current.bm_accu)
			inc_mark_hit = "[+]";
		else
			inc_mark_hit = "";

		best.bm_accu = max(best.bm_accu, current.bm_accu);

		logout.record() << "[Result]\tAccurarcy = " << current.bm_accu
			<< ", \tBest = " << best.bm_accu << inc_mark_hit;
	}

	virtual void result()
	{
		logout.record() << "[Task]\tEntity Answering";
		logout.record() << "[Result]\tAccurarcy = " << best.bm_accu;
	}
};