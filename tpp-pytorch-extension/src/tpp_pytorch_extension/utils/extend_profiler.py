###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

try:
    from torch.autograd.profiler_util import *
    from torch.autograd.profiler import profile
except ImportError:
    from torch.autograd.profiler import *


def el_nested_key_averages(self, only_top_level):
    """Averages all function events over their keys.

    Returns:
        An EventList containing FunctionEventAvg objects.
    """
    stats = defaultdict(FunctionEventAvg)
    for evt in self:
        if only_top_level and getattr(evt, "cpu_parent", None):
            continue
        stats[evt.nested_key] += evt
    if only_top_level:
        for evt in stats.values():
            evt.self_cpu_time_total = evt.cpu_time_total
        return EventList(stats.values())
    else:
        top_level_stats = []
        child_list = defaultdict(list)
        for k, evt in stats.items():
            evt.key = k
            evt.input_shapes = ""
            nested_keys = k.split(".")
            if len(nested_keys) == 1:
                top_level_stats.append(evt)
            else:
                parent = ".".join(nested_keys[:-1])
                child_list[parent].append(evt)

        top_level_stats = sorted(
            top_level_stats,
            key=lambda evt: getattr(evt, "cpu_time_total"),
            reverse=True,
        )
        for evt in stats.values():
            evt.children = list(
                sorted(
                    child_list[evt.key],
                    key=lambda evt1: getattr(evt1, "cpu_time_total"),
                    reverse=True,
                )
            )

        def traverse(lst, evt):
            lst.append(evt)
            for e in evt.children:
                traverse(lst, e)

        lst = []
        for evt in top_level_stats:
            traverse(lst, evt)
        return EventList(lst)


def p_nested_key_averages(self, only_top_level=False):
    self._check_finish()
    return self.function_events.nested_key_averages(only_top_level)


def fe_append_cpu_child(self, child):
    """Append a CPU child of type FunctionEvent.

    One is supposed to append only dirrect children to the event to have
    correct self cpu time being reported.
    """
    assert isinstance(child, FunctionEvent)
    self.cpu_children.append(child)
    child.cpu_parent = self


def fe_nested_key(self):
    plist = [self.name]
    nested_name = getattr(self, "nested_name", None)
    if nested_name:
        return nested_name
    p = getattr(self, "cpu_parent", None)
    while p:
        plist.insert(0, p.name)
        p = getattr(p, "cpu_parent", None)
    nested_name = ".".join(plist)
    return nested_name


def print_op_timings(prof, use_gpu=False, prefix="prof"):
    def get_interval(e):
        if hasattr(e, "cpu_interval"):
            return e.cpu_interval
        # if hasattr(e, 'time_range'): return e.time_range
        return e.time_range

    sorted_fe = sorted(
        prof.function_events,
        key=lambda event: [get_interval(event).start, -get_interval(event).end],
    )
    start_time = get_interval(sorted_fe[0]).start if len(sorted_fe) > 0 else 0
    with open("%s.OpTimings.txt" % prefix, "w") as f:
        for i, fe in enumerate(sorted_fe):
            fe_name = getattr(fe, "nested_key", fe.name)
            cstr = ""
            if use_gpu:
                for kinfo in fe.kernels:
                    cstr += " %10.3f %10.3f %8.3f %8.3f " % (
                        (kinfo.interval.start - start_time) / 1000.0,
                        (kinfo.interval.end - start_time) / 1000.0,
                        (kinfo.interval.start - get_interval(fe).start) / 1000.0,
                        kinfo.interval.elapsed_us() / 1000.0,
                    )
            print(
                "%-6d %6d %12.4f %12.4f %12.4f %2s %s   %-40s    %s"
                % (
                    i,
                    fe.id,
                    (get_interval(fe).start - start_time) / 1000.0,
                    (get_interval(fe).end - start_time) / 1000.0,
                    get_interval(fe).elapsed_us() / 1000.0,
                    fe.thread,
                    cstr,
                    fe_name.replace(" ", "_"),
                    fe.input_shapes,
                ),
                file=f,
            )


if not hasattr(FunctionEvent, "set_cpu_parent"):
    FunctionEvent.append_cpu_child = fe_append_cpu_child
FunctionEvent.nested_key = property(fe_nested_key)
EventList.nested_key_averages = el_nested_key_averages
profile.nested_key_averages = p_nested_key_averages
profile.nested_key_averages.__doc__ = EventList.nested_key_averages.__doc__
profile.print_op_timings = print_op_timings
