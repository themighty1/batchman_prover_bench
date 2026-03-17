The main improvement of batchman was that - The prover only needs to commit to / evaluate the active branch, not all B branches. This is indeed the core computational 
  saving.         

  3. The prover then runs a (small) VOLE-based ZK proof that the product of all B tokens is zero: ∏ v_i = 0. This proves that  
  at least one branch's topology is satisfied — but it's a product-is-zero check, not a "MAC == 0" check on a single branch. 


However the cost of that zero product proof protocol dominates the bandwidth cost of the protocol. COncretely even for the smallest possible ISA size of 50 ops, to prove a single execution step would require sending 100 extension field elements of 64 bits.

so 100*8 = 800 bytes


We are targetting the baseline upload bandwidth of 10MPBS, it becomes clear that the zero product protocol alone will not allow proving speeds higher than 1.2/800 = 15K step per second.


So the workaround that we can, the obvious workaround is to not use the zero product protocol, but instead allow the prover to prove that one of its branches evaluates to zero using some other means. The problem here is that, you know, this essentially boils down to a set membership proof, which can be done with a polynomial commitment scheme. The problem is that the proof can be carried out in order to prove the set, the prover must know the entire set, which brings us back to having the prover evaluate every single branch, including the non-active ones. So essentially with that step, we are just undoing the benefits that the batchmen gives us, which was not avoiding the need for the prover to evaluate every single branch. And yet, it was worth benchmarking if there are some gains to be had. An additional constraint is that, so essentially what we're gonna boil down the protocol to is the prover will know all the message authentication codes for all the branches, and he needs to prove in zero knowledge that one of those message authentication codes is zero. How can he do it?

To use a more high-level language of the paper, the prover has a token for each branch. Only for the active branch, his token is zero. His token for all other branches can be any random number. So the original zero product protocol in it, the prover proved that one of his tokens is zero by proving the product of all tokens. Obviously, if at least one element is zero, the product would be zero.


But when we switch to a set membership proof, there is a problem. In order to use a polynomial commitment scheme, the prover has to commit first. He needs to commit to the quotient polynomial, but in order to do that, he needs to have the original full set. The prover is unable to learn the entire set at the time of the protocol execution. He can only learn the set after he commits to all his inputs of the protocol. So what we do is, after the prover is fully committed to all the values, the verifier reveals to him its delta, and the prover can regenerate all the zero tokens locally to create the entire set. But before the verifier revealed the delta, the prover actually committed to his active tokens, so the verifier already holds that commitment. Now what remains at the end of the protocol is for the prover to prove that his zero tokens, his subset that he committed to, is contained within the entire set. So Bineos provides a simple API for that.


Yeah, so because the set membership proof must be run at the end of the main batchment protocol, it inevitably adds additional latency to the entire protocol. It cannot be interleaved, it cannot be run during the protocols because the delta has to remain hidden from the prover throughout the entire protocol. So BNS is the state-of-the-art ZK proving system which natively works with binary fields. This is exactly what we need. Unfortunately, given the state-of-the-art proving speed, leave much to be desired.
