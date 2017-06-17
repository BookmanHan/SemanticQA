#pragma once
#include "Import.hpp"
#include "Model.hpp"
#include "Solver.hpp"

class JointMFactorUnit
{
public:
	struct 
	{
		vector<vec> rep_entity;
		vector<vec> rep_relation_subj;
		vector<vec> rep_relation_obj;
	} _p, _derv_x, _derv_grad;

public:
	AscentAdaDeltaPositive& solver;

public:
	const int		dim;
	const double	sigma;

public:
	JointMFactorUnit(int dim, int entity_count, int relation_count, int word_count, double sigma, AscentAdaDeltaPositive& solver)
		: dim(dim), sigma(sigma), solver(solver)
	{
		_p.rep_entity.resize(entity_count);
		for_each(_p.rep_entity.begin(), _p.rep_entity.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		_p.rep_relation_subj.resize(relation_count);
		for_each(_p.rep_relation_subj.begin(), _p.rep_relation_subj.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		_p.rep_relation_obj.resize(relation_count);
		for_each(_p.rep_relation_obj.begin(), _p.rep_relation_obj.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		_derv_x.rep_entity.resize(entity_count);
		for_each(_derv_x.rep_entity.begin(), _derv_x.rep_entity.end(),
			[=](vec& elem){elem = zeros(dim); });

		_derv_x.rep_relation_subj.resize(relation_count);
		for_each(_derv_x.rep_relation_subj.begin(), _derv_x.rep_relation_subj.end(),
			[=](vec& elem){elem = zeros(dim); });

		_derv_x.rep_relation_obj.resize(relation_count);
		for_each(_derv_x.rep_relation_obj.begin(), _derv_x.rep_relation_obj.end(),
			[=](vec& elem){elem = zeros(dim); });

		_derv_grad.rep_entity.resize(entity_count);
		for_each(_derv_grad.rep_entity.begin(), _derv_grad.rep_entity.end(),
			[=](vec& elem){elem = zeros(dim); });

		_derv_grad.rep_relation_subj.resize(relation_count);
		for_each(_derv_grad.rep_relation_subj.begin(), _derv_grad.rep_relation_subj.end(),
			[=](vec& elem){elem = zeros(dim); });

		_derv_grad.rep_relation_obj.resize(relation_count);
		for_each(_derv_grad.rep_relation_obj.begin(), _derv_grad.rep_relation_obj.end(),
			[=](vec& elem){elem = zeros(dim); });
	}

public:
	void save(FormatFile & file)
	{
		file << _p.rep_entity << _p.rep_relation_subj << _p.rep_relation_obj;
		file << _derv_x.rep_entity << _derv_x.rep_relation_subj << _derv_x.rep_relation_obj;
		file << _derv_grad.rep_entity << _derv_grad.rep_relation_subj << _derv_grad.rep_relation_obj;
	}

	void load(FormatFile & file)
	{
		file >> _p.rep_entity >> _p.rep_relation_subj >> _p.rep_relation_obj;
		file >> _derv_x.rep_entity >> _derv_x.rep_relation_subj >> _derv_x.rep_relation_obj;
		file >> _derv_grad.rep_entity >> _derv_grad.rep_relation_subj >> _derv_grad.rep_relation_obj;
	}

public:
	double prob(const tuple<int, int, int>& triplet)
	{
		vec& head = _p.rep_entity[get<0>(triplet)];
		vec& tail = _p.rep_entity[get<2>(triplet)];
		vec& relation_head = _p.rep_relation_subj[get<1>(triplet)];
		vec& relation_tail = _p.rep_relation_obj[get<1>(triplet)];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;

		double score = log(sum(head_feature % tail_feature)) * sigma
			- sum(abs(head_feature - tail_feature));

		return score;
	}

	void train_knowledge
		(const tuple<int, int, int>& triplet, const double alpha)
	{
		vec& head = _p.rep_entity[get<0>(triplet)];
		vec& tail = _p.rep_entity[get<2>(triplet)];
		vec& relation_head = _p.rep_relation_subj[get<1>(triplet)];
		vec& relation_tail = _p.rep_relation_obj[get<1>(triplet)];

		vec head_feature = head % relation_head;
		vec tail_feature = tail % relation_tail;
		vec feature = head_feature % tail_feature;
		vec grad = sign(head_feature - tail_feature);

		double sum_feature = sum(feature) + 1e-50;

		vec grad_head =
			- alpha * grad % head_feature / head
			+ alpha * feature / head / sum_feature * sigma;
		vec grad_relation_head =
			- alpha * grad % head_feature / relation_head
			+ alpha * feature / relation_head / sum_feature * sigma;
		vec grad_tail =
			alpha * grad % tail_feature / tail
			+ alpha * feature / tail / sum_feature * sigma;
		vec grad_relation_tail =
			alpha * grad % tail_feature / relation_tail
			+ alpha * feature / relation_tail / sum_feature * sigma;

		solver.gradient(
			_derv_grad.rep_entity[get<0>(triplet)],
			_derv_x.rep_entity[get<0>(triplet)],
			_p.rep_entity[get<0>(triplet)],
			grad_head);

		solver.gradient(
			_derv_grad.rep_entity[get<2>(triplet)],
			_derv_x.rep_entity[get<2>(triplet)],
			_p.rep_entity[get<2>(triplet)],
			grad_tail);

		solver.gradient(
			_derv_grad.rep_relation_subj[get<1>(triplet)],
			_derv_x.rep_relation_subj[get<1>(triplet)],
			_p.rep_relation_subj[get<1>(triplet)],
			grad_relation_head);

		solver.gradient(
			_derv_grad.rep_relation_obj[get<1>(triplet)],
			_derv_x.rep_relation_obj[get<1>(triplet)],
			_p.rep_relation_obj[get<1>(triplet)],
			grad_relation_tail);
	}
};

class JointMFactor
	:public Model
{
protected:
	vector<JointMFactorUnit*>	factors;
	vector<vec> rep_entity;
	vector<vec> rep_relation_subj;
	vector<vec> rep_relation_obj;

public:
	AscentAdaDeltaPositive& solver;

public:
	const double	margin;
	const int		dim;
	const int		n_factor;
	const double	sigma;

public:
	JointMFactor(
		const DataSet& ds,
		int n_factor,
		int dim,
		double margin,
		double sigma,
		AscentAdaDeltaPositive& solver)
		:Model(ds), margin(margin), dim(dim), n_factor(n_factor), sigma(sigma), solver(solver)
	{
		logout.record() << "[Name]\tJoint.Multiple.FactorE";
		logout.record() << "[Dimension]\t" << dim;
		logout.record() << "[Sigma]\t" << sigma;
		logout.record() << "[Training Threshold]\t" << margin;
		logout.record() << "[Factor Number]\t" << n_factor;

		for (auto i = 0; i < n_factor; ++i)
		{
			factors.push_back(new JointMFactorUnit(dim, dm.n_entity + 10, dm.n_relation + 10, dm.n_word + 200, sigma, solver));
		}
	}

public:
	virtual double probability(const tuple<int, int, int>& datum) override
	{
		vec score(n_factor);
		auto i_score = score.begin();
		for (auto factor : factors)
		{
			*i_score++ = factor->prob(datum);
		}

		return sum(score);
	}

	virtual void train_knowledge(const tuple<int, int, int>& datum) override
	{
		tuple<int, int, int> triplet_false = dm.negtive_sample(datum);

		if (probability(datum) - probability(triplet_false) > margin)
			return;

		for (auto factor : factors)
		{
			factor->train_knowledge(datum, 1.0);
			factor->train_knowledge(triplet_false, -1.0);
		}
	}

	virtual void save(FormatFile & file) override
	{
		rep_entity.clear();
		for (auto i = 0; i < dm.n_entity; ++i)
		{
			vec rep;
			for (auto factor : factors)
			{
				rep = join_cols(rep, factor->_p.rep_entity[i]);
			}
			rep_entity.push_back(rep);
		}

		rep_relation_subj.clear();
		for (auto i = 0; i < dm.n_relation; ++i)
		{
			vec rep;
			for (auto factor : factors)
			{
				rep = join_cols(rep, factor->_p.rep_relation_subj[i]);
			}
			rep_relation_subj.push_back(rep);
		}

		rep_relation_obj.clear();
		for (auto i = 0; i < dm.n_relation; ++i)
		{
			vec rep;
			for (auto factor : factors)
			{
				rep = join_cols(rep, factor->_p.rep_relation_obj[i]);
			}
			rep_relation_obj.push_back(rep);
		}

		file << rep_entity;
		file << rep_relation_subj;
		file << rep_relation_obj;

		for (auto factor : factors)
		{
			factor->save(file);
		}
	}

	virtual void load(FormatFile & file) override
	{
		file >> rep_entity;
		file >> rep_relation_subj;
		file >> rep_relation_obj;

		for (auto factor : factors)
		{
			factor->load(file);
		}
	}
};
