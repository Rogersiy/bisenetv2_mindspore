import time
import mindspore as ms
from . import logger
from .common import save_checkpoint


class Callback(ms.Callback):
    def __init__(self, cfg, network, optimizer, eval_dataset=None):
        super(Callback, self).__init__()
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.epoch_start = time.time()
        if self.cfg.run_eval:
            self.eval_dataset = eval_dataset
        if self.cfg.run_profilor:
            self.profiler = ms.Profiler(start_profile=False)

    def on_train_begin(self, run_context):
        logger.info("Start Training")

    def on_train_epoch_begin(self, run_context):
        self.epoch_start = time.time()
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if self.cfg.run_profilor and epoch_num == 3:
            self.profiler.start()

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = (cb_params.get("cur_epoch_num", 1) // self.cfg.print_pre_epoch) + 1
        cur_step = cb_params.get("cur_epoch_num", 1) * self.cfg.log_interval
        loss, cond, scaling_sens = cb_params.net_outputs
        step = self.optimizer.global_step
        if self.optimizer.dynamic_lr:
            cur_lr = self.optimizer.learning_rate(step - 1)[0].asnumpy()
        else:
            cur_lr = self.optimizer.learning_rate.asnumpy()
        logger.info(f"Epoch {cur_epoch_num}/{self.cfg.epochs}, "
                    f"step {cur_step % self.cfg.steps_per_epoch}/{self.cfg.steps_per_epoch}, "
                    f"loss: {loss.asnumpy():.4f}, "
                    f"cond: {cond.asnumpy()}, "
                    f"scaling_sens: {scaling_sens.asnumpy()}, "
                    f"cur_lr: {cur_lr:.4f}, "
                    f"cost {((time.time() - self.epoch_start) / cb_params.batch_num) * 1000:.2f} ms")
        epoch_num = cb_params.cur_epoch_num
        if self.cfg.run_profilor and epoch_num == 5:
            self.profiler.stop()
            self.profiler.analyse()
            run_context.request_stop()
        if cb_params.get("cur_epoch_num", 1) % self.cfg.print_pre_epoch == 0:
            save_checkpoint(self.cfg, self.network, self.optimizer, cur_epoch_num)
            # if self.cfg.run_eval and cur_epoch_num > self.cfg.epochs // 2:
            #     run_eval(self.cfg, self.network, self.eval_dataset, cur_epoch_num, cb_params.batch_num)
