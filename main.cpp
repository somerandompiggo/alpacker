#include <iostream>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// llama.cpp
#include <common/common.h>
#include <llama.h>

// bot
#include <dpp/dpp.h>

const std::string botToken = "your token here";

const char* MODEL_LOCATION = "yourmodel.gguf";

const int n_len = 2048; // sequence length in tokens including prompt

gpt_params params;
llama_context_params ctx_params;
std::vector<llama_token> tokens_list;
llama_batch batch;

llama_model * model;
llama_context * ctx;

const std::string discriminator = "alpaca, ";

// chat
std::string chatQuery(std::string userPrompt) {
    params.prompt = "\n\n### Instruction: Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Input:\n\n" + userPrompt + "\n\n### Response:\n\n";

    ctx_params = llama_context_default_params();

    ctx_params.seed  = 12354;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        printf("Fatal error: failed to create the llama context\n");
    }

    const int n_ctx = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        printf("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        printf("%s:        either reduce n_parallel or increase n_ctx\n", __func__);
    }

    tokens_list = ::llama_tokenize(ctx, params.prompt, true);

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
    }

    batch = llama_batch_init(512, 0, 1);
    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        printf("Fatal error: llama_decode() failed\n");
    }

    std::string output;

    // main loop
    int n_cur = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_len) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of stream?
            if (new_token_id == llama_token_eos(model) || n_cur == n_len) {
                printf("\n\nEnd of stream token detected, breaking.\n");
                break;
            }

            output += llama_token_to_piece(ctx, new_token_id);
            printf("%s", llama_token_to_piece(ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        }
    }
    return output;
}

int initialize() {
    llama_backend_init(params.numa); // init LLM
    llama_model_params model_params = llama_model_default_params(); // init model
    // model_params.n_gpu_layers = 99; gpu layer loading here
    model = llama_load_model_from_file(MODEL_LOCATION, model_params);

    if (model == NULL) {
        printf("Fatal error: could not load model\n");
        return -1;
    }
    return 0;
}

int main() {
    initialize();

    dpp::cluster bot(botToken, dpp::i_default_intents | dpp::i_message_content);

    bot.on_log(dpp::utility::cout_logger());

    bot.on_message_create([&bot](const dpp::message_create_t& event) {

        if (event.msg.content.substr(0,discriminator.length()) == discriminator) {
            std::string query = event.msg.content.substr(discriminator.length(), event.msg.content.length());
            std::string response = chatQuery(query);

            event.reply(response);
            std::cout << response << std::endl << query << std::endl;
        }
    });

    bot.start(dpp::st_wait);
}









// prediction

//int main() {
//    gpt_params params;
//
//    const int n_len = 1024; // sequence length in tokens including prompt
//
//    params.prompt = prompt;
//
//    llama_backend_init(params.numa); // init LLM
//    llama_model_params model_params = llama_model_default_params(); // init model
//    // model_params.n_gpu_layers = 99; gpu layer loading here
//    llama_model * model = llama_load_model_from_file(MODEL_LOCATION, model_params);
//
//    if (model == NULL) {
//        printf("Fatal error: could not load model\n");
//        return -1;
//    }
//
//    llama_context_params ctx_params = llama_context_default_params();
//
//    ctx_params.seed  = 1234;
//    ctx_params.n_ctx = 2048;
//    ctx_params.n_threads = params.n_threads;
//    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
//
//    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
//
//    if (ctx == NULL) {
//        printf("Fatal error: failed to create the llama context\n");
//        return -1;
//    }
//
//    std::vector<llama_token> tokens_list;
//    tokens_list = ::llama_tokenize(ctx, params.prompt, true);
//
//    const int n_ctx = llama_n_ctx(ctx);
//    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());
//
//    // make sure the KV cache is big enough to hold all the prompt and generated tokens
//    if (n_kv_req > n_ctx) {
//        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
//        LOG_TEE("%s:        either reduce n_parallel or increase n_ctx\n", __func__);
//        return 1;
//    }
//
//    for (auto id : tokens_list) {
//        fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
//    }
//
//    llama_batch batch = llama_batch_init(512, 0, 1);
//    // evaluate the initial prompt
//    for (size_t i = 0; i < tokens_list.size(); i++) {
//        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
//    }
//
//    // llama_decode will output logits only for the last token of the prompt
//    batch.logits[batch.n_tokens - 1] = true;
//
//    if (llama_decode(ctx, batch) != 0) {
//        printf("Fatal error: llama_decode() failed\n");
//        return -1;
//    }
//
//    // main loop
//    int n_cur = batch.n_tokens;
//    int n_decode = 0;
//
//    const auto t_main_start = ggml_time_us();
//
//    while (n_cur <= n_len) {
//        // sample the next token
//        {
//            auto   n_vocab = llama_n_vocab(model);
//            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);
//
//            std::vector<llama_token_data> candidates;
//            candidates.reserve(n_vocab);
//
//            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
//                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
//            }
//
//            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
//
//            // sample the most likely token
//            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);
//
//            // is it an end of stream?
//            if (new_token_id == llama_token_eos(model) || n_cur == n_len) {
//                printf("\n\nEnd of stream token detected, breaking.\n");
//                break;
//            }
//
//            printf("%s", llama_token_to_piece(ctx, new_token_id).c_str());
//            fflush(stdout);
//
//            // prepare the next batch
//            llama_batch_clear(batch);
//
//            // push this new token for next evaluation
//            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
//
//            n_decode += 1;
//        }
//
//        n_cur += 1;
//
//        // evaluate the current batch with the transformer model
//        if (llama_decode(ctx, batch)) {
//            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
//            return 1;
//        }
//    }
//}