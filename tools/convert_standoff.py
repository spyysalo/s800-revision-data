#!/usr/bin/env python3

import sys


DEFAULT_ENCODING = 'utf-8'


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-e', '--encoding', default=DEFAULT_ENCODING,
                    help='text encoding (default {})'.format(DEFAULT_ENCODING))
    ap.add_argument('-i', '--ignore-errors', default=False, action='store_true',
                    help='ignore format errors')
    ap.add_argument('ann', nargs='+', help='annotation')
    return ap


########## annotation ##########


class FormatError(Exception):
    pass


class SpanDeleted(Exception):
    pass


class Annotation(object):
    def __init__(self, id_, type_):
        self.id_ = id_
        self.type_ = type_
        self.normalizations = []
        self.notes = []

    def resolve_references(self, ann_by_id):
        pass

    def add_normalization(self, ann):
        self.normalizations.append(ann)

    def add_note(self, ann):
        self.notes.append(ann)


class Textbound(Annotation):
    def __init__(self, id_, type_, offsets, text):
        Annotation.__init__(self, id_, type_)
        self.text = text
        self.offsets = []
        for start_end in offsets.split(';'):
            start, end = start_end.split()
            self.offsets.append((int(start), int(end)))


class XMLElement(Textbound):
    def __init__(self, id_, type_, offsets, text, attributes):
        Textbound.__init__(self, id_, type_, offsets, text)
        self.attributes = attributes


class ArgAnnotation(Annotation):
    def __init__(self, id_, type_, args):
        Annotation.__init__(self, id_, type_)
        self.args = args

    def resolve_references(self, ann_by_id):
        raise NotImplementedError()    # TODO


class Relation(ArgAnnotation):
    def __init__(self, id_, type_, args):
        ArgAnnotation.__init__(self, id_, type_, args)


class Event(ArgAnnotation):
    def __init__(self, id_, type_, trigger, args):
        ArgAnnotation.__init__(self, id_, type_, args)
        self.trigger = trigger


class Attribute(Annotation):
    def __init__(self, id_, type_, target_id, value):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.value = value
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]


class Normalization(Annotation):
    def __init__(self, id_, type_, target_id, ref, reftext):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.ref = ref
        self.reftext = reftext
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]
        self.target.add_normalization(self)


class Equiv(Annotation):
    def __init__(self, id_, type_, targets):
        Annotation.__init__(self, id_, type_)
        self.targets = targets

    def resolve_references(self, ann_by_id):
        raise NotImplementedError()    # TODO


class Note(Annotation):
    def __init__(self, id_, type_, target_id, text):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.text = text
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]
        self.target.add_note(self)


def parse_xml(fields):
    id_, type_offsets, text, attributes = fields
    type_offsets = type_offsets.split(' ')
    type_, offsets = type_offsets[0], type_offsets[1:]
    return XMLElement(id_, type_, offsets, text, attributes)


def parse_textbound(fields):
    id_, type_offsets, text = fields
    type_offsets = type_offsets.split(' ', 1)
    type_, offsets = type_offsets
    return Textbound(id_, type_, offsets, text)


def parse_relation(fields):
    # allow a variant where the two initial TAB-separated fields are
    # followed by an extra tab
    if len(fields) == 3 and not fields[2]:
        fields = fields[:2]
    id_, type_args = fields
    type_args = type_args.split(' ')
    type_, args = type_args[0], type_args[1:]
    return Relation(id_, type_, args)


def parse_event(fields):
    id_, type_trigger_args = fields
    type_trigger_args = type_trigger_args.split(' ')
    type_trigger, args = type_trigger_args[0], type_trigger_args[1:]
    type_, trigger = type_trigger.split(':')
    return Event(id_, type_, trigger, args)


def parse_attribute(fields):
    id_, type_target_value = fields
    type_target_value = type_target_value.split(' ')
    if len(type_target_value) == 3:
        type_, target, value = type_target_value
    else:
        type_, target = type_target_value
        value = None
    return Attribute(id_, type_, target, value)


def parse_normalization(fields):
    if len(fields) == 3:
        id_, type_target_ref, reftext = fields
    elif len(fields) == 2:    # Allow missing reference text
        id_, type_target_ref = fields
        reftext = ''
    type_, target, ref = type_target_ref.split(' ')
    return Normalization(id_, type_, target, ref, reftext)


def parse_note(fields):
    id_, type_target, text = fields
    type_, target = type_target.split(' ')
    return Note(id_, type_, target, text)


def parse_equiv(fields):
    id_, type_targets = fields
    type_targets = type_targets.split(' ')
    type_, targets = type_targets[0], type_targets[1:]
    return Equiv(id_, type_, targets)


parse_standoff_func = {
    'T': parse_textbound,
    'R': parse_relation,
    'E': parse_event,
    'N': parse_normalization,
    'M': parse_attribute,
    'A': parse_attribute,
    'X': parse_xml,
    '#': parse_note,
    '*': parse_equiv,
}


def parse_standoff_line(l, ln, fn, options):
    try:
        return parse_standoff_func[l[0]](l.split('\t'))
    except Exception:
        if options.ignore_errors:
            error('failed to parse line {} in {}: {}'.format(ln, fn, l))
            return None
        else:
            raise FormatError('error on line {} in {}: {}'.format(ln, fn, l))


def parse_ann_file(fn, options):
    annotations = []
    with open(fn, 'r', encoding=options.encoding) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if not l or l.isspace():
                continue
            ann = parse_standoff_line(l, ln, fn, options)
            if ann is not None:
                annotations.append(ann)
    return annotations


def resolve_references(annotations, options):
    ann_by_id = {}
    for a in annotations:
        ann_by_id[a.id_] = a
    for a in annotations:
        a.resolve_references(ann_by_id)


def normalize_space(string):
    return ' '.join(string.split())


def output(annotations, options):
    textbounds = [a for a in annotations if isinstance(a, Textbound)]
    textbounds.sort(key=lambda t:(t.offsets[0][0], -t.offsets[-1][1]))
    for t in textbounds:
        refs = [n.ref.replace('Taxonomy:', '') for n in t.normalizations]
        notes = [normalize_space(n.text) for n in t.notes]
        if not refs and t.type_ == 'Out-of-scope':
            continue
        if not refs:
            refs = ['0']
        print('{}\t{}'.format(','.join(refs), t.text))
        #print('{}\t{}\t{}\t{}'.format(
        #    ','.join(refs), t.text, t.type_, ';'.join(notes)))


def main(argv):
    args = argparser().parse_args(argv[1:])
    for fn in args.ann:
        annotations = parse_ann_file(fn, args)
        resolve_references(annotations, args)
        output(annotations, args)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

