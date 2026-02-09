#ifndef ZK_PROVER_H__
#define ZK_PROVER_H__
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include <iostream>
#include "emp-zk/emp-zk-bool/ostriple.h"
#include "emp-zk/emp-zk-bool/zk_bool_circuit_exec.h"
#include "emp-zk/emp-zk-bool/polynomial.h"

template<typename IO, typename COTType = FerretCOT<IO>>
class ZKBoolCircExecPrv:public ZKBoolCircExec<IO, COTType> { public:
	using ZKBoolCircExec<IO, COTType>::pub_label;
	ZKBoolCircExecPrv() {
		PRG prg(fix_key);
		prg.random_block(pub_label, 2);
		pub_label[0] = pub_label[0] & makeBlock(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL);
		pub_label[1] = pub_label[1] & makeBlock(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFEULL);
		pub_label[1] = pub_label[1] ^ makeBlock(0, 1);
	}
	block not_gate(const block&a) override {
		return a ^ makeBlock(0, 1);
	}
	template<typename T, typename C>
	void set_ostriple(OSTriple<T, C> *ostriple) {
		this->ostriple = ostriple;
	}
};



template<typename IO, typename COTType = FerretCOT<IO>>
class ZKProver: public ProtocolExecution {
public:
	IO* io = nullptr;
	OSTriple<IO, COTType> *ostriple = nullptr;
	PolyProof<IO, COTType> *polyproof = nullptr;
	ZKBoolCircExecPrv<IO, COTType> *gen = nullptr;
	ZKProver(IO** ios, int threads, ZKBoolCircExecPrv<IO, COTType> *t, void * state): ProtocolExecution(ALICE) {
		this->io = ios[0];
		this->gen = t;
		ostriple = new OSTriple<IO, COTType>(ALICE, threads, ios, state);
		polyproof = new PolyProof<IO, COTType>(ALICE, ios[0], ostriple->ferret);
		t->template set_ostriple<IO, COTType>(ostriple);
		t->polyproof = this->polyproof;
	}
	~ZKProver() {
		delete polyproof;
		delete ostriple;
	}

	/* 
	 * Prover is the receiver in iterative COT
	 * interface: get 1 authenticated bit
	 * authenticated message, last bit as the value
	 * embeded in label
	 */
	void feed(block * label, int party, const bool* b, int length) {
		if(party == ALICE)
			ostriple->authenticated_bits_input(label, b, length);
		else if (party == PUBLIC) {
			for(int i = 0; i < length; ++i)
				label[i] = gen->public_label(b[i]);
		}
	}

	/*
	 * check correctness of triples using cut and choose and bucketing
	 * check correctness of the output
	 */
	void reveal(bool* b, int party, const block * label, int length) {
		if(party == ALICE) {
			for(int i = 0; i < length; ++i)
				b[i] = getLSB(label[i]);
		}
		else if (party == BOB or party == PUBLIC) {
			ostriple->verify_output(b, label, length);
		}
	}
};
#endif //ZK_PROVER_H__
