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
		vector<vec> rep_word;
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

		_p.rep_word.resize(word_count); 
		for_each(_p.rep_word.begin(), _p.rep_word.end(),
			[=](vec& elem){elem = normalise(randu(dim), 2); });

		_derv_x.rep_entity.resize(entity_count);
		for_each(_derv_x.rep_entity.begin(), _derv_x.rep_entity.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_x.rep_relation_subj.resize(relation_count);
		for_each(_derv_x.rep_relation_subj.begin(), _derv_x.rep_relation_subj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_x.rep_relation_obj.resize(relation_count);
		for_each(_derv_x.rep_relation_obj.begin(), _derv_x.rep_relation_obj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_x.rep_word.resize(word_count);
		for_each(_derv_x.rep_word.begin(), _derv_x.rep_word.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_entity.resize(entity_count);
		for_each(_derv_grad.rep_entity.begin(), _derv_grad.rep_entity.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_relation_subj.resize(relation_count);
		for_each(_derv_grad.rep_relation_subj.begin(), _derv_grad.rep_relation_subj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_relation_obj.resize(relation_count);
		for_each(_derv_grad.rep_relation_obj.begin(), _derv_grad.rep_relation_obj.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });

		_derv_grad.rep_word.resize(word_count);
		for_each(_derv_grad.rep_word.begin(), _derv_grad.rep_word.end(),
			[=](vec& elem){elem = normalise(zeros(dim), 2); });
	}

public:
	void save(FormatFile & file)
	{
		file << _p.rep_entity << _p.rep_relation_subj << _p.rep_relation_obj << _p.rep_word;
		file << _derv_x.rep_entity << _derv_x.rep_relation_subj << _derv_x.rep_relation_obj << _derv_x.rep_word;
		file << _derv_grad.rep_entity << _derv_grad.rep_relation_subj << _derv_grad.rep_relation_obj << _derv_grad.rep_word;
	}

	void load(FormatFile & file)
	{
		file >> _p.rep_entity >> _p.rep_relation_subj >> _p.rep_relation_obj >> _p.rep_word;
		file >> _derv_x.rep_entity >> _derv_x.rep_relation_subj >> _derv_x.rep_relation_obj >> _derv_x.rep_word;
		file >> _derv_grad.rep_entity >> _derv_grad.rep_relation_subj >> _derv_grad.rep_relation_obj >> _derv_grad.rep_word;
	}

public:
	vec prob_des(const vector<int>& entity_des)
	{
		vec result = ones(dim);
		for (auto i = entity_des.begin(); i != entity_des.end(); ++i)
		{
			result %= _p.rep_word[*i];
		}

		return result;
	}

	double prob(const tuple<int, int, int>& triplet, const vector<int>& head_des, const vector<int>& tail_des)
	{
		vec& head = _p.rep_entity[get<0>(triplet)];
		vec& tail = _p.rep_entity[get<2>(triplet)];
		vec& relation_head = _p.rep_relation_subj[get<1>(triplet)];
		vec& relation_tail = _p.rep_relation_obj[get<1>(triplet)];
		vec rep_head = prob_des(head_des);
		vec rep_tail = prob_des(tail_des);

		rep_head /= sum(rep_head);
		rep_tail /= sum(rep_tail);

		vec head_feature = head % relation_head % rep_head;
		vec tail_feature = tail % relation_tail % rep_tail;

		return log(sum(head_feature % tail_feature)) * sigma
			- sum(abs(head_feature - tail_feature));
	}

	void train(const tuple<int, int, int>& triplet, const vector<int>& head_des, const vector<int>& tail_des, const double alpha)
	{
		vec& head = _p.rep_entity[get<0>(triplet)];
		vec& tail = _p.rep_entity[get<2>(triplet)];
		vec& relation_head = _p.rep_relation_subj[get<1>(triplet)];
		vec& relation_tail = _p.rep_relation_obj[get<1>(triplet)];
		vec rep_head_p = prob_des(head_des);
		vec rep_tail_p = prob_des(tail_des);
		vec rep_head = rep_head_p / sum(rep_head_p);
		vec rep_tail = rep_tail_p / sum(rep_tail_p);

		vec head_feature = head % relation_head % rep_head;
		vec tail_feature = tail % relation_tail % rep_tail;
		vec feature = head_feature % tail_feature;
		vec grad = sign(head_feature - tail_feature);

		vec grad_head = -alpha * grad % head_feature / head
			+ alpha * feature / head / sum(feature) * sigma;
		vec grad_relation_head = -alpha * grad % head_feature / relation_head
			+ alpha * feature / relation_head / sum(feature) * sigma;
		vec grad_tail = alpha * grad % tail_feature / tail
			+ alpha * feature / tail / sum(feature) * sigma;
		vec grad_relation_tail = alpha * grad % tail_feature / relation_tail
			+ alpha * feature / relation_tail / sum(feature) * sigma;

		vec grad_word_head = -alpha * grad % head_feature / rep_head
			+ alpha * feature / rep_head / sum(feature) * sigma;
		vec grad_word_tail = alpha * grad % tail_feature / rep_tail
			+ alpha * feature / rep_tail / sum(feature) * sigma;

		for (auto i = head_des.begin(); i != head_des.end(); ++i)
		{
			vec grad_word = normalise(
				grad_word_head % rep_head % (1.0/_p.rep_word[*i] - sum(rep_head_p/_p.rep_word[*i])/sum(rep_head_p)));
			solver.gradient(
				_derv_grad.rep_word[*i], 
				_derv_x.rep_word[*i], 
				_p.rep_word[*i], 
				grad_word);
		}

		for (auto i = tail_des.begin(); i != tail_des.end(); ++i)
		{
			vec grad_word = normalise(
				grad_word_tail % rep_tail % (1.0 / _p.rep_word[*i] - sum(rep_tail_p / _p.rep_word[*i]) / sum(rep_tail_p)));
			solver.gradient(
				_derv_grad.rep_word[*i], 
				_derv_x.rep_word[*i], 
				_p.rep_word[*i],
				grad_word);
		}

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

public:
	AscentAdaDeltaPositive& solver;

public:
	const double	margin;
	const double	alpha;
	const int		dim;
	const int		n_factor;
	const double	sigma;

public:
	JointMFactor(
		const DataSet& ds,
		int n_factor,
		int dim,
		double alpha,
		double margin,
		double sigma,
		AscentAdaDeltaPositive& solver)
		:Model(ds), margin(margin), alpha(alpha), dim(dim), n_factor(n_factor), sigma(sigma), solver(solver)
	{
		logout.record() << "[Name]\tJoint.Multiple.FactorE";
		logout.record() << "[Dimension]\t" << dim;
		logout.record() << "[Learning Rate]\t" << alpha;
		logout.record() << "[Sigma]\t" << sigma;
		logout.record() << "[Training Threshold]\t" << margin;
		logout.record() << "[Factor Number]\t" << n_factor;

		for (auto i = 0; i < n_factor; ++i)
		{
			factors.push_back(new JointMFactorUnit(dim, dm.n_entity + 10, dm.n_relation + 10, dm.n_word + 200, sigma, solver));
		}
	}

public:
	virtual double probability(const tuple<int, int, int>& datum, const vector<int>& des_head, const vector<int>& des_tail) override
	{
		vec score(n_factor);
		auto i_score = score.begin();
		for (auto factor : factors)
		{
			*i_score++ = factor->prob(datum, des_head, des_tail);
		}

		return sum(score);
	}

	virtual void train_once(const tuple<int, int, int>& datum, const vector<int>& des_head, const vector<int>& des_tail) override
	{
		tuple<int, int, int> triplet_f = datum;
		if (randu() > 0.5)
		{
			get<0>(triplet_f) = rand() % dm.n_entity;
		}
		else
		{
			get<2>(triplet_f) = rand() % dm.n_entity;
		}

		if (probability(datum, des_head, des_tail)-probability(triplet_f, des_head, des_tail) > margin)
			return;

		for (auto i = 0; i < n_factor; ++i)
		{
			factors[i]->train(datum, des_head, des_tail, alpha);
			factors[i]->train(triplet_f, des_head, des_tail, - alpha);
		}
	}

	virtual void save(FormatFile & file) override
	{
		for (auto factor : factors)
		{
			factor->save(file);
		}
	}

	virtual void load(FormatFile & file) override
	{
		for (auto factor : factors)
		{
			factor->load(file);
		}
	}

	virtual int infer_entity(const vector<int>& des) override
	{
		vec prob_ent = ones(dm.n_entity);

		for (auto factor : factors)
		{
			vec rep_des = factor->prob_des(des);
			rep_des /= sum(rep_des);
			for (auto i = 0; i < dm.n_entity; ++i)
			{
				prob_ent[i] *= as_scalar(rep_des.t() * factor->_p.rep_entity[i]);
			}
		}

		return prob_ent.index_max();
	}

	virtual int infer_relation(const vector<int>& des) override
	{
		return 0;
	}
};
