from tqdm import tqdm


class ProgressBar(tqdm):
    def __init__(
        self,
        episodes: int,
        batches: int,
        sequences_per_batch: int,
        prefix: str = None,
        suffix: str = None,
        leave: bool = None,
    ):
        self.episodes = episodes
        self.curr_ep = 1
        self.batches = batches
        self.curr_b = 1
        self.sequences_per_batch = sequences_per_batch
        self.curr_s = 1
        self.prefix = prefix
        self.suffix = suffix
        self.total_iterations = episodes * batches * sequences_per_batch
        super().__init__(
            total=self.total_iterations,
            leave=leave,
            desc=self.gen_description(),
        )

    def gen_description(self):
        start = f"{self.prefix}|" if self.prefix else ""
        end = f"|{self.suffix}" if self.suffix else ""
        return f"{start}Episode {self.curr_ep}/{self.episodes}|Batch {self.curr_b}/{self.batches}|Sequence {self.curr_s}/{self.sequences_per_batch}{end}"

    def update(self, ep: int = None, batch: int = None, seq: int = None):
        diff_e = (
            ((ep or self.curr_ep) - self.curr_ep)
            * self.batches
            * self.sequences_per_batch
        )
        diff_b = ((batch or self.curr_b) - self.curr_b) * self.sequences_per_batch
        diff_s = (seq or self.curr_s) - self.curr_s
        super().update(diff_e + diff_b + diff_s)

        super().set_description
        self.curr_ep = ep or self.curr_ep
        self.curr_b = batch or self.curr_b
        self.curr_s = seq or self.curr_s
        super().set_description(self.gen_description())
