open Graph_def
open Names
open Tactician_ltac1_record_plugin
open Ltac_plugin

module K = Graph_api.Make(Capnp.BytesMessage)

type graph_state = { node_index : int; edge_index : int }

module CapnpGraphWriter(P : sig type path end)(G : GraphMonadType with type node = P.path * int) = struct

  let nt2nt transformer (nt : G.node node_type) cnt =
    let open K.Builder.Graph.Node.Label in
    match nt with
    | Root -> root_set cnt
    | ContextDef id -> context_def_set cnt (Id.to_string id)
    | ContextAssum id -> context_assum_set cnt (Id.to_string id)
    | Definition { previous; def_type; path } ->
      let cdef = definition_init cnt in
      let open K.Builder.Definition in
      (* TODO: We should probably derive the hash of a constant form its path (until we obtain actual hashes) *)
      name_set cdef (Libnames.string_of_path path);
      (match def_type with
       | TacticalConstant (c, proof) ->
         let hash = Constant.UserOrd.hash c in
         hash_set cdef (Stdint.Uint64.of_int hash);
         let tconst = tactical_constant_init cdef in
         let arr = TacticalConstant.tactical_proof_init tconst (List.length proof) in
         List.iteri (fun i ({ tactic; base_tactic; interm_tactic; tactic_hash; arguments; tactic_exact
                            ; root; context; ps_string }
                            : G.node tactical_step) ->
                      let arri = Capnp.Array.get arr i in
                      let state = K.Builder.ProofStep.state_init arri in
                      let capnp_tactic = K.Builder.ProofStep.tactic_init arri in
                      K.Builder.ProofState.root_set_int_exn state @@ snd root;
                      let _ = K.Builder.ProofState.context_set_list state
                          (List.map (fun x -> Stdint.Uint32.of_int @@ snd x) context) in
                      K.Builder.ProofState.text_set state ps_string;
                      K.Builder.Tactic.ident_set_int_exn capnp_tactic tactic_hash;
                      K.Builder.Tactic.exact_set capnp_tactic tactic_exact;
                      K.Builder.Tactic.text_set capnp_tactic
                        (Pp.string_of_ppcmds @@ Sexpr.format_oneline (Pptactic.pr_glob_tactic (Global.env ()) tactic));
                      K.Builder.Tactic.base_text_set capnp_tactic
                        (Pp.string_of_ppcmds @@ Sexpr.format_oneline (Pptactic.pr_glob_tactic (Global.env ()) base_tactic));
                      K.Builder.Tactic.interm_text_set capnp_tactic
                        (Pp.string_of_ppcmds @@ Sexpr.format_oneline (Pptactic.pr_glob_tactic (Global.env ()) interm_tactic));
                      let arg_arr = K.Builder.Tactic.arguments_init capnp_tactic (List.length arguments) in
                      List.iteri (fun i arg ->
                          let arri = Capnp.Array.get arg_arr i in
                          match arg with
                          | None -> K.Builder.Tactic.Argument.unresolvable_set arri
                          | Some (dep, index) ->
                            let node = K.Builder.Tactic.Argument.term_init arri in
                            K.Builder.Tactic.Argument.Term.dep_index_set_exn node @@ transformer dep;
                            K.Builder.Tactic.Argument.Term.node_index_set_int_exn node index
                        ) arguments;
                      ()
                    ) proof;
       | ManualConst c ->
         let hash = Constant.UserOrd.hash c in
         hash_set cdef (Stdint.Uint64.of_int hash);
         manual_constant_set cdef
       | Ind (m, i) ->
         let hash = Hashtbl.hash (i, MutInd.UserOrd.hash m) in
         hash_set cdef (Stdint.Uint64.of_int hash);
         inductive_set cdef
       | Construct ((m, i), j) ->
         let hash = Hashtbl.hash (i, j, MutInd.UserOrd.hash m) in
         hash_set cdef (Stdint.Uint64.of_int @@ hash);
         constructor_set cdef
       | Proj p ->
         let hash = Projection.Repr.UserOrd.hash p in
         hash_set cdef @@ Stdint.Uint64.of_int hash; (* TODO: This is probably not a good hash *)
         projection_set cdef
      )
    | ConstEmpty -> const_empty_set cnt
    | SortSProp -> sort_s_prop_set cnt
    | SortProp -> sort_prop_set cnt
    | SortSet -> sort_set_set cnt
    | SortType -> sort_type_set cnt
    | Rel -> rel_set cnt
    | Evar i ->
      let p = evar_init cnt in
      K.Builder.IntP.value_set p @@ Stdint.Uint64.of_int i
    | EvarSubst -> evar_subst_set cnt
    | Cast -> cast_set cnt
    | Prod _ -> prod_set cnt
    | Lambda _ -> lambda_set cnt
    | LetIn _ -> let_in_set cnt
    | App -> app_set cnt
    | AppFun -> app_fun_set cnt
    | AppArg -> app_arg_set cnt
    | Case -> case_set cnt
    | CaseBranch -> case_branch_set cnt
    | Fix -> fix_set cnt
    | FixFun _ -> fix_fun_set cnt
    | CoFix -> co_fix_set cnt
    | CoFixFun _ -> co_fix_fun_set cnt
    | Int i ->
      let p = int_init cnt in
      K.Builder.IntP.value_set p @@ Stdint.Uint64.of_int @@ snd @@ Uint63.to_int2 i
    | Float f ->
      let p = float_init cnt in
      K.Builder.FloatP.value_set p @@ float64_to_float f
    | Primitive p -> primitive_set cnt (CPrimitives.to_string p)
  let et2et (et : edge_type) =
    let open K.Builder.EdgeClassification in
    match et with
    | ContextElem -> ContextElem
    | ContextSubject -> ContextSubject
    | ContextDefType -> ContextDefType
    | ContextDefTerm -> ContextDefTerm
    | ConstType -> ConstType
    | ConstUndef -> ConstUndef
    | ConstDef -> ConstDef
    | ConstOpaqueDef -> ConstOpaqueDef
    | ConstPrimitive -> ConstPrimitive
    | IndType -> IndType
    | IndConstruct -> IndConstruct
    | ProjTerm -> ProjTerm
    | ConstructTerm -> ConstructTerm
    | CastTerm -> CastTerm
    | CastType -> CastType
    | ProdType -> ProdType
    | ProdTerm -> ProdTerm
    | LambdaType -> LambdaType
    | LambdaTerm -> LambdaTerm
    | LetInDef -> LetInDef
    | LetInType -> LetInType
    | LetInTerm -> LetInTerm
    | AppFunPointer -> AppFunPointer
    | AppFunValue -> AppFunValue
    | AppArgPointer -> AppArgPointer
    | AppArgValue -> AppArgValue
    | AppArgOrder -> AppArgOrder
    | CaseTerm -> CaseTerm
    | CaseReturn -> CaseReturn
    | CaseBranchPointer -> CaseBranchPointer
    | CaseInd -> CaseInd
    | CBConstruct -> CBConstruct
    | CBTerm -> CBTerm
    | FixMutual -> FixMutual
    | FixReturn -> FixReturn
    | FixFunType -> FixFunType
    | FixFunTerm -> FixFunTerm
    | CoFixMutual -> CoFixMutual
    | CoFixReturn -> CoFixReturn
    | CoFixFunType -> CoFixFunType
    | CoFixFunTerm -> CoFixFunTerm
    | RelPointer -> RelPointer
    | EvarSubstPointer -> EvarSubstPointer
    | EvarSubstOrder -> EvarSubstOrder
    | EvarSubstValue -> EvarSubstValue

  let write_graph capnp_graph transformer
      node_count edge_count builder =
    let nodes = K.Builder.Graph.nodes_init capnp_graph node_count in
    let edges = K.Builder.Graph.edges_init capnp_graph edge_count in
    let state = { node_index = node_count - 1; edge_index = 0 } in
    let arrays_add { node_index; edge_index } label children =
      let node = Capnp.Array.get nodes node_index in
      nt2nt transformer label @@ K.Builder.Graph.Node.label_init node;
      let cc = List.length children in
      K.Builder.Graph.Node.children_count_set_exn node cc;
      K.Builder.Graph.Node.children_index_set_int_exn node (if cc = 0 then 0 else edge_index);
      let edge_index = List.fold_left (fun ei (label, (tp, ti)) ->
          let et = Capnp.Array.get edges ei in
          K.Builder.Graph.EdgeTarget.label_set et @@ et2et label;
          let ctarget = K.Builder.Graph.EdgeTarget.target_init et in
          K.Builder.Graph.EdgeTarget.Target.dep_index_set_exn ctarget @@ transformer tp;
          K.Builder.Graph.EdgeTarget.Target.node_index_set_int_exn ctarget @@ ti;
          ei + 1
        ) edge_index children in
      { node_index = node_index - 1; edge_index } in
    ignore(builder state arrays_add)
end
