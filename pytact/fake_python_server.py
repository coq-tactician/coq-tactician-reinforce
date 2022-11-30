import sys
import socket
import argparse
import pytact.graph_visualize as gv
import capnp
import pytact.graph_api_capnp as graph_api_capnp
from pytact.data_reader import online_definitions_initialize, online_data_predict
from pytact.graph_api_capnp_cython import PredictionProtocol_Request_Initialize_Reader, PredictionProtocol_Request_Predict_Reader, PredictionProtocol_Request_CheckAlignment_Reader

def text_prediction_loop(r, s):
    tactics = [ 'idtac "is it working?"', 'idtac "yes it is working!"', 'auto' ]
    for g in r:
        msg_type = g.which()
        if msg_type == "predict":
            print(g.predict.state.text)
            preds = [
                {'tacticText': t,
                 'confidence': 0.5} for t in tactics ]
            response = graph_api_capnp.PredictionProtocol.Response.new_message(textPrediction=preds)
            print(response)
            response.write_packed(s)
            import time
            time.sleep(1)
        elif msg_type == "synchronize":
            print(g)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(synchronized=g.synchronize)
            print(response)
            response.write_packed(s)
        elif msg_type == "initialize":
            print(g)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(initialized=None)
            print(response)
            response.write_packed(s)
        else:
            raise Exception("Capnp protocol error")

def prediction_loop(definitions, tactics, incoming_messages, capnp_socket):
    for msg in incoming_messages:
        msg_type = msg.which()
        if msg_type == "predict":
            with online_data_predict(
                    definitions,
                    PredictionProtocol_Request_Predict_Reader(msg.predict)) as proof_state:
                gv.visualize_proof_state(proof_state)
                singleArgs = [t.ident for t in tactics if t.parameters == 0]
                preds = [{'tactic': {'ident': t}, 'arguments': [], 'confidence': 0.5} for t in singleArgs]
                if len(proof_state.context) > 0:
                    oneArg = [t.ident for t in tactics if t.parameters == 1]
                    hyp_node = proof_state.context[0]
                    preds2 = [
                        {'tactic': {'ident': t},
                         'arguments': [{'term' : {'depIndex': 0, 'nodeIndex': hyp_node}}],
                         'confidence': 0.5} for t in oneArg ]
                    preds += preds2
                for d in definitions.definitions:
                    if d.name == "Coq.Init.Logic.I":
                        oneArg = [t.ident for t in tactics if t.parameters == 1]
                        preds2 = [
                            {'tactic': {'ident': t },
                             'arguments': [{'term' : {'depIndex': 1, 'nodeIndex': d.node.nodeid}}],
                             'confidence': 0.5} for t in oneArg ]
                        preds += preds2
                response = graph_api_capnp.PredictionProtocol.Response.new_message(prediction=preds)
                print(response)
                response.write_packed(capnp_socket)
                import time
                time.sleep(1)
        else:
            return msg

def graph_initialize_loop(incoming_messages, capnp_socket):
    msg = next(incoming_messages, None)
    while msg is not None:
        msg_type = msg.which()
        if msg_type == "predict":
            raise Exception('Predict message received without a preceding initialize message')
        elif msg_type == "synchronize":
            print(msg)
            response = graph_api_capnp.PredictionProtocol.Response.new_message(synchronized=msg.synchronize)
            print(response)
            response.write_packed(capnp_socket)
            msg = next(incoming_messages, None)
        elif msg_type == "checkAlignment":
            cython_check_alignment = PredictionProtocol_Request_CheckAlignment_Reader(msg.checkAlignment)
            with online_definitions_initialize(
                    cython_check_alignment.graph,
                    cython_check_alignment.representative) as definitions:
                for cluster in definitions.clustered_definitions:
                    print('cluster:')
                    for d in cluster:
                        print(f'    {d.name}')
                for t in cython_check_alignment.tactics:
                    print(t)
                alignment = {'unalignedTactics': [ t.ident for t in cython_check_alignment.tactics],
                             'unalignedDefinitions': [d.node.nodeid for d in definitions.definitions]}
                response = graph_api_capnp.PredictionProtocol.Response.new_message(alignment=alignment)
                response.write_packed(capnp_socket)
                msg = next(incoming_messages, None)
        elif msg_type == "initialize":
            print('---------------- New prediction context -----------------')
            cython_init = PredictionProtocol_Request_Initialize_Reader(msg.initialize)
            with online_definitions_initialize(cython_init.graph, cython_init.representative) as definitions:
                for cluster in definitions.clustered_definitions:
                    print('cluster:')
                    for d in cluster:
                        print(f'    {d.name}')
                for t in cython_init.tactics:
                    print(t)
                print(cython_init.log_annotation)
                response = graph_api_capnp.PredictionProtocol.Response.new_message(initialized=None)
                print(response)
                response.write_packed(capnp_socket)
                msg = prediction_loop(definitions, cython_init.tactics, incoming_messages, capnp_socket)
        else:
            raise Exception("Capnp protocol error")
        import time
        time.sleep(1)

def run_session(args, capnp_socket):
    incoming_messages = graph_api_capnp.PredictionProtocol.Request.read_multiple_packed(
        capnp_socket,
        traversal_limit_in_words=2**64-1)
    if args.mode == 'text':
        print('Python server running in text mode')
        text_prediction_loop(incoming_messages, capnp_socket)
    elif args.mode == 'graph':
        print('Python server running in graph mode')
        graph_initialize_loop(incoming_messages, capnp_socket)
    else:
        raise Exception("The 'mode' argument needs to be either 'text' or 'graph'")

def main():
    parser = argparse.ArgumentParser(
        description = 'Example python server capable of communicating with Coq',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('mode',
                        type=str,
                        help='"graph" to communicate in graph-mode, "text" to communicate in text-mode')
    parser.add_argument('--tcp',
                        type = int,
                        default = 0,
                        help='Run in tcp mode instead of stdin mode on the specified port.')
    args = parser.parse_args()

    if args.tcp != 0:
        addr = ('localhost', args.tcp)
        server_sock = socket.create_server(addr)
        try:
            while True:
                capnp_socket, remote_addr = server_sock.accept()
                print(f"coq client connected {remote_addr}")
                run_session(args, capnp_socket)
        finally:
            print(f'closing the server on port {addr[1]}')
            server_sock.close()
    else:
        capnp_socket = socket.socket(fileno=sys.stdin.fileno())
        run_session(args, capnp_socket)

if __name__ == '__main__':
    main()
